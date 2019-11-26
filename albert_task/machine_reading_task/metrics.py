"""
定义各类性能指标
"""
import re
import json
import math
import collections

from collections import OrderedDict


def get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    # 位置和分数组成的元组构成的列表
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def compute_softmax(scores):
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


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, output_prediction_file=None, output_nbest_file=None):
    """Write final predictions to the json file and log-odds of null if needed."""

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature["example_index"]].append(feature)

    unique_id_to_result = {}  # 将result的结果读取出来存成字典
    for result in all_results:
        unique_id_to_result[result["unique_id"]] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]  # 获取每个example下的features

        prelim_predictions = []

        for (feature_index, feature) in enumerate(features):  # 取出example下每个feature
            result = unique_id_to_result[feature["unique_id"]]  # 取出每个feature预测出来的结果
            start_indexes = get_best_indexes(result["start_logits"], n_best_size)  # 取出n个分数最高的index
            end_indexes = get_best_indexes(result["end_logits"], n_best_size)  # 取出n个分数最高的index

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

        probs = compute_softmax(total_scores)  # 对所有的分数计算交叉熵

        nbest_json = []
        for (i, entry) in enumerate(nbest):  # 保存每个example的每个回答和对应的分数，以及分数比例
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example["qid"]] = nbest_json[0]["text"]  # 存储分数最大的回答结果

        all_nbest_json[example["qid"]] = nbest_json  # 将所有的example按照id：nbest_json的字典形式存储

    with open(output_prediction_file, "w", encoding="utf8") as fw:
        json.dump(all_predictions, fw, indent=4, ensure_ascii=False)

    with open(output_nbest_file, "w", encoding="utf8") as fw:
        json.dump(all_nbest_json, fw, indent=4, ensure_ascii=False)


def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def split_char(context):
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


def remove_punctuation(text):
    """
    去除标点符号
    :param text:
    :return:
    """
    text = str(text).lower().strip()
    punc_char = {'-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
                 '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
                 '「', '」', '（', '）', '－', '～', '『', '』'}
    new_text = []
    for char in text:
        if char in punc_char:
            continue
        else:
            new_text.append(char)
    return "".join(new_text)


def find_lcs(s1, s2):
    """
    最长公共子序列作为真正例
    :param s1:
    :param s2:
    :return:
    """
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    m_max = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > m_max:
                    m_max = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - m_max:p], m_max


def evaluate(original_data, prediction_data):
    """
    对验证集进行评估
    :param original_data:
    :param prediction_data:
    :return:
    """
    f1 = 0.0
    em = 0.0
    total_count = 0
    skip_count = 0
    for instance in original_data["data"]:
        for para in instance["paragraphs"]:
            for qas in para['qas']:
                total_count += 1
                query_id = qas['id'].strip()
                # 在验证集中一个问题提供了多个答案
                answers = [x["text"] for x in qas['answers']]

                if query_id not in prediction_data:
                    print('Unanswered question: {}\n'.format(query_id))
                    skip_count += 1
                    continue

                prediction = str(prediction_data[query_id])
                f1 += calc_f1_score(answers, prediction)
                em += calc_em_score(answers, prediction)

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score


def calc_f1_score(answers, prediction):
    """
    计算f1值，最长公共子序作为真正例
    :param answers:
    :param prediction:
    :return:
    """
    f1_scores = []
    for ans in answers:
        ans_segs = split_char(ans)
        prediction_segs = split_char(prediction)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    """
    计算em值
    :param answers:
    :param prediction:
    :return:
    """
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def get_eval(original_file, prediction_file):
    """
    得到预测的性能指标
    :param original_file:
    :param prediction_file:
    :return:
    """
    with open(original_file, 'r', encoding="utf8") as fr:
        original_data = json.load(fr)

    with open(prediction_file, "r", encoding="utf8") as fr:
        prediction_data = json.load(fr)

    f1, em = evaluate(original_data, prediction_data)
    average = (f1 + em) * 0.5
    res = dict(average=average, f1=f1, em=em)
    return res


