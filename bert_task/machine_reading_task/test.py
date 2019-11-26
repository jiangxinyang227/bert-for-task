import os
import json
import random
import collections
import sys

sys.path.append(os.path.dirname(os.getcwd()))

from bert import tokenization


def improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                        orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def split_char(context):
    new_context = []
    pre_is_digit = False
    pre_is_letter = False
    for char in context:
        if "0" <= char <= "9":
            if pre_is_digit:
                new_context[-1] += char
            else:
                new_context.append(char)
                pre_is_digit = True
                pre_is_letter = False
        elif "a" <= char <= "z" or "A" <= char <= "Z":

            if pre_is_letter:
                new_context[-1] += char
            else:
                new_context.append(char)
                pre_is_letter = True
                pre_is_digit = False
        else:
            new_context.append(char)
            pre_is_digit = False
            pre_is_letter = False
    return new_context


def read_data(file_path, is_training):
    """
    :param file_path:
    :param is_training:
    :return:
    """
    with open(file_path, 'r', encoding="utf8") as f:
        train_data = json.load(f)
        train_data = train_data['data']

    examples = []
    # 1, 遍历所有的训练数据，取出每一篇文章
    for article in train_data:
        # 2， 遍历每一篇文章，取出该文章下的所有段落
        for para in article['paragraphs']:
            id_ = para["id"]
            context = para['context']  # 取出当前段落的内容
            doc_tokens = split_char(context)

            # char_to_word_offset的长度等于context的长度，但是列表中的最大值为len(doc_tokens) - 1
            # 主要作用是为了维护doc_tokens中的token的位置对应到在context中的位置
            char_to_word_offset = []
            for index, token in enumerate(doc_tokens):
                for i in range(len(token)):
                    char_to_word_offset.append(index)

            # 把问答对读取出来
            for qas in para['qas']:
                qid = qas['id']
                ques_text = qas['question']
                ans_text = qas['answers'][0]['text']

                start_position_final = -1
                end_position_final = -1
                if is_training:

                    # 取出在原始context中的start和end position
                    start_position = qas['answers'][0]['answer_start']

                    # 按照答案长度取去计算结束位置
                    end_position = start_position + len(ans_text) - 1

                    # 如果在start的位置上是对应原始context中的空字符，则往上加一位
                    while context[start_position] == " " or context[start_position] == "\t" or \
                            context[start_position] == "\r" or context[start_position] == "\n":
                        start_position += 1

                    # 从context中start和end的位置映射到doc_tokens中的位置

                    start_position_final = char_to_word_offset[start_position]
                    end_position_final = char_to_word_offset[end_position]

                    if doc_tokens[start_position_final] in {"。", "，", "：", ":", ".", ","}:
                        start_position_final += 1

                examples.append({'doc_tokens': doc_tokens,
                                 'orig_answer_text': context,
                                 'qid': qid,
                                 'question': ques_text,
                                 'answer': ans_text,
                                 'start_position': start_position_final,
                                 'end_position': end_position_final})

    return examples


def trans_to_features(examples, is_training):
    """
    将输入转化为索引表示
    :param examples: 输入
    :param is_training:
    :return:
    """
    vocab_file = "../bert_model/chinese_L-12_H-768_A-12/vocab.txt"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    features = []
    unique_id = 1000000000
    for (example_index, example) in enumerate(examples):
        # 用wordpiece的方法对query进行分词处理
        query_tokens = tokenizer.tokenize(example['question'])
        # 给定query一个最大长度来控制query的长度
        if len(query_tokens) > 64:
            query_tokens = query_tokens[: 64]

        # 主要是针对context构造索引，之前我们将中文，标点符号，空格，一连串的数字，英文单词分割存储在doc_tokens中
        # 但在bert的分词器中会将一连串的数字，中文，英文等分割成子词，也就是说经过bert的分词之后得到的tokens和之前
        # 获得的doc_tokens是不一样的，因此我们仍需要对start和end position从doc_tokens中的位置映射到当前tokens的位置
        tok_to_orig_index = []  # 存储未分词的token的索引，但长度和下面的相等
        orig_to_tok_index = []  # 存储分词后的token的索引，但索引不是连续的，会存在跳跃的情况
        all_doc_tokens = []  # 存储分词后的token，理论上长度是要大于all_tokens的

        for (i, token) in enumerate(example['doc_tokens']):
            sub_tokens = tokenizer.tokenize(token)
            # orig_to_tok_index的长度等于doc_tokens，里面每个值存储的是doc_tokens中的token在all_doc_tokens中的起止索引值
            # 用来将在all_token中的start和end转移到all_doc_tokens中
            orig_to_tok_index.append([len(all_doc_tokens)])
            for sub_token in sub_tokens:
                # tok_to_orig_index的长度等于all_doc_tokens, 里面会有重复的值
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
            orig_to_tok_index[-1].append(len(all_doc_tokens) - 1)

        tok_start_position = -1
        tok_end_position = -1
        if is_training:
            tok_start_position = orig_to_tok_index[example['start_position']][0]  # 原来token到新token的映射，这是新token的起点
            tok_end_position = orig_to_tok_index[example['end_position']][1]

            tok_start_position, tok_end_position = improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example['orig_answer_text'])

        max_length = 512
        doc_stride = 128
        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_length - len(query_tokens) - 3

        doc_spans = []
        _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])

        # 在使用bert的时候，一般会将最大的序列长度控制在512，因此对于长度大于最大长度的context，我们需要将其分成多个片段
        # 采用滑窗的方式，滑窗大小是小于最大长度的，因此分割的片段之间是存在重复的子片段。
        start_offset = 0  # 截取的片段的起始位置
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset

            # 当长度超标，需要使用滑窗
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):  # 当length < max_len时，该条件成立
                break
            start_offset += min(length, doc_stride)

        # 组合query和context的片段成一个序列输入到bert中
        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            # 因为片段之间会存在重复的子片段，但是子片段中的token在不同的片段中的重要性是不一样的，
            # 在这里根据上下文的数量来决定token的重要性，在之后预测时对于出现在两个片段中的token，只取重要性高的片段
            # 中的token的分数作为该token的分数
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[
                    split_token_index]  # 映射当前span组成的句子对的索引到原始token的索引

                # 在利用滑窗分割多个span时会存在有的词出现在两个span中，但最后统计的时候，我们只能选择一个span，因此
                # 作者根据该词上下文词的数量构建了一个分数，取分数最高的那个span
                is_max_context = check_is_max_context(doc_spans, doc_span_index, split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
    #
    #         assert len(input_ids) == self.__max_length
    #         assert len(input_mask) == self.__max_length
    #         assert len(segment_ids) == self.__max_length

            start_position = -1
            end_position = -1
            if is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                if tok_start_position == -1 and tok_end_position == -1:
                    start_position = 0  # 问题本来没答案，0是[CLS]的位子
                    end_position = 0
                else:  # 如果原本是有答案的，那么去除没有答案的feature
                    out_of_span = False
                    doc_start = doc_span.start  # 映射回原文的起点和终点
                    doc_end = doc_span.start + doc_span.length - 1

                    if not (tok_start_position >= doc_start and tok_end_position <= doc_end):  # 该划窗没答案作为无答案增强
                        out_of_span = True
                    if out_of_span:
                        start_position = 0
                        end_position = 0
                    else:
                        doc_offset = len(query_tokens) + 2
                        start_position = tok_start_position - doc_start + doc_offset
                        end_position = tok_end_position - doc_start + doc_offset

            features.append({'unique_id': unique_id,
                             'example_index': example_index,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': start_position,
                             'end_position': end_position})
            unique_id += 1
    return features


# train_file = "data/cmrc2018/cmrc2018_train.json"
# examples = read_data(train_file, True)
# trans_to_features(examples, True)


# tok_to_orig_index = []  # 存储未分词的token的索引，但长度和下面的相等
# orig_to_tok_index = []  # 存储分词后的token的索引，但索引不是连续的，会存在跳跃的情况
# all_doc_tokens = []  # 存储分词后的token，理论上长度是要大于all_tokens的
#
#
# vocab_file = "../bert_model/chinese_L-12_H-768_A-12/vocab.txt"
# tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
#
# tokens = ["你", "是", "breaking", "mohamed", "fayed", "哈"]
# for (i, token) in enumerate(tokens):
#     sub_tokens = tokenizer.tokenize(token)
#     # orig_to_tok_index的长度等于doc_tokens，里面每个值存储的是doc_tokens中的token在all_doc_tokens中的起止索引值
#     # 用来将在all_token中的start和end转移到all_doc_tokens中
#     orig_to_tok_index.append([len(all_doc_tokens)])
#     for sub_token in sub_tokens:
#         # tok_to_orig_index的长度等于all_doc_tokens, 里面会有重复的值
#         tok_to_orig_index.append(i)
#         all_doc_tokens.append(sub_token)
#     orig_to_tok_index[-1].append(len(all_doc_tokens) - 1)
#
# print(orig_to_tok_index)
# print(all_doc_tokens)