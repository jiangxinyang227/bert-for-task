import json
import os
import sys
import collections
import math

sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import tensorflow as tf
from model import BertMachineReading
from bert import tokenization


class Predictor(object):
    def __init__(self, config):
        self.model = None
        self.config = config

        self.vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.query_length = config["query_length"]
        self.doc_stride = config["doc_stride"]
        self.max_length = config["max_length"]
        self.max_answer_length = config["max_answer_length"]
        self.n_best_size = config["n_best_size"]

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def _improve_answer_span(self, doc_tokens, input_start, input_end, tokenizer,
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

    def _check_is_max_context(self, doc_spans, cur_span_index, position):
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

    def _split_char(self, context):
        """
        将原文分割成列表返回，主要是确保一连串的数字，或者英文单词作为一个单独的token存在
        :param context:
        :return:
        """
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

    def read_data(self, query, context):
        """
        处理输入的问题
        :param query: 输入的问题
        :param context: 存在答案的上下文
        :return:
        """

        doc_tokens = self._split_char(context)

        example = {'doc_tokens': doc_tokens,
                   'orig_answer_text': context,
                   'question': query,
                   'start_position': -1,
                   'end_position': -1}

        return example

    def trans_to_features(self, example):
        """
        将输入转化为索引表示
        :param example: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)
        features = []
        unique_id = 1000000000

        query_tokens = tokenizer.tokenize(example['question'])
        # 给定query一个最大长度来控制query的长度
        if len(query_tokens) > self.query_length:
            query_tokens = query_tokens[: self.query_length]

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

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = self.max_length - len(query_tokens) - 3

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
            start_offset += min(length, self.doc_stride)

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
                is_max_context = self._check_is_max_context(doc_spans, doc_span_index, split_token_index)
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
            while len(input_ids) < self.max_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == self.max_length
            assert len(input_mask) == self.max_length
            assert len(segment_ids) == self.max_length

            features.append({'unique_id': unique_id,
                             'doc_span_index': doc_span_index,
                             'tokens': tokens,
                             'token_to_orig_map': token_to_orig_map,
                             'token_is_max_context': token_is_max_context,
                             'input_ids': input_ids,
                             'input_mask': input_mask,
                             'segment_ids': segment_ids,
                             'start_position': -1,
                             'end_position': -1})
            unique_id += 1
        return features

    def sentence_to_ids(self, features):
        unique_id = []
        input_ids = []
        input_masks = []
        segment_ids = []
        start_position = []
        end_position = []
        for feature in features:
            unique_id.append(feature["unique_id"])
            input_ids.append(feature["input_ids"])
            input_masks.append(feature["input_mask"])
            segment_ids.append(feature["segment_ids"])
            start_position.append(feature["start_position"])
            end_position.append(feature["end_position"])

        return dict(unique_id=unique_id,
                    input_ids=input_ids,
                    input_masks=input_masks,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position)

    def get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        # 位置和分数组成的元组构成的列表
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def get_predictions(self, example, features, results, n_best_size, max_answer_length):
        """Write final predictions to the json file and log-odds of null if needed."""

        # example_index_to_features = collections.defaultdict(list)
        # for feature in features:
        #     example_index_to_features[feature["example_index"]].append(feature)

        unique_id_to_result = {}  # 将result的结果读取出来存成字典
        for result in results:
            unique_id_to_result[result["unique_id"]] = result

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):  # 取出example下每个feature
            result = unique_id_to_result[feature["unique_id"]]  # 取出每个feature预测出来的结果
            start_indexes = self.get_best_indexes(result["start_logits"], n_best_size)  # 取出n个分数最高的index
            end_indexes = self.get_best_indexes(result["end_logits"], n_best_size)  # 取出n个分数最高的index

            # 根据一些设定条件去除一些无效的start_index 和 end_index
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature["tokens"]):
                        continue
                    if end_index >= len(feature["tokens"]):
                        continue
                    if start_index not in feature["token_to_orig_map"]:
                        continue
                    if end_index not in feature["token_to_orig_map"]:
                        continue
                    if not feature["token_is_max_context"].get(start_index, False):  # 如果start不是分数最大的span，则跳过
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue

                    # 得到一个example下所有feature中得到的预测结果
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result["start_logits"][start_index],  # start的分数
                            end_logit=result["end_logits"][end_index]))  # end的分数

        # 按照start+end的分数排序
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        # 存储一个example中的n个预测结果
        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:  # 存储nbest个最好的结果
                break
            feature = features[pred.feature_index]  # 根据feature_index索引到相应的feature
            tok_tokens = feature["tokens"][pred.start_index:(pred.end_index + 1)]  # 取出每个feature下对应的预测tokens
            orig_doc_start = feature["token_to_orig_map"][pred.start_index]  # 取出原始doc_tokens中的开始位置
            orig_doc_end = feature["token_to_orig_map"][pred.end_index]  # 取出原始doc_tokens中的结束位置
            orig_tokens = example["doc_tokens"][orig_doc_start:(orig_doc_end + 1)]  # 取出原始doc_tokens中对应的预测tokens
            tok_text = "".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = "".join(tok_text.split())
            orig_text = "".join(orig_tokens)

            seen_predictions[tok_text] = True

            nbest.append(
                _NbestPrediction(
                    text=orig_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None  # 保存分数最大的非空回答
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = self.compute_softmax(total_scores)  # 对所有的分数计算交叉熵

        nbest_json = []
        for (i, entry) in enumerate(nbest):  # 保存每个example的每个回答和对应的分数，以及分数比例
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        return nbest_json[0]["text"]

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))

    def create_model(self):
        """
                根据config文件选择对应的模型，并初始化
                :return:
                """
        self.model = BertMachineReading(config=self.config, is_training=False)

    def predict(self, query, context):
        """
        给定问句和上下文，给定其相应的回答
        :param query:
        :param context:
        :return:
        """
        example = self.read_data(query, context)
        features = self.trans_to_features(example)
        input_data = self.sentence_to_ids(features)

        start_logits, end_logits = self.model.infer(self.sess, input_data)
        results = []
        for unique_id, start_logit, end_logit in zip(input_data["unique_id"],
                                                     start_logits,
                                                     end_logits):
            results.append(dict(unique_id=unique_id,
                                start_logits=start_logit.tolist(),
                                end_logits=end_logit.tolist()))
        answer = self.get_predictions(example, features, results, self.n_best_size, self.max_answer_length)
        return answer
