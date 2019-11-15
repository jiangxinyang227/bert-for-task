import os
import json
import random
import sys
from itertools import chain

sys.path.append(os.path.dirname(os.getcwd()))

from bert import tokenization


class TrainData(object):
    def __init__(self, config):

        self.__vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.__output_path = config["output_path"]
        if not os.path.exists(self.__output_path):
            os.makedirs(self.__output_path)
        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长
        self._batch_size = config["batch_size"]

    @staticmethod
    def read_data(file_path):
        """
        读取数据
        :param file_path:
        :return: 返回分词后的文本内容和标签，inputs = [], labels = []
        """
        inputs = []
        labels = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    text, label = line.strip().split("<SEP>")
                    inputs.append(text.strip().split(" "))
                    labels.append(label.strip().split(" "))
                except:
                    continue

        return inputs, labels

    def trans_to_index(self, inputs, labels):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :param labels: 输出
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []
        new_labels = []

        for text, label in zip(inputs, labels):

            tokens = []
            new_label = []
            for token, tag in zip(text, label):
                token = tokenizer.tokenize(token)
                tokens.extend(token)
                new_label.extend([tag] * len(token))

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_id = tokenizer.convert_tokens_to_ids(tokens)

            label = ["o"] + label + ["o"]

            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))
            new_labels.append(label)

        return input_ids, input_masks, segment_ids, new_labels

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_ids = [[label_to_index[item] for item in label] for label in labels]
        return labels_ids

    def padding(self, input_ids, input_masks, segment_ids, label_ids, label_to_index):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids
        :param label_to_index
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len = [], [], [], [], []
        for input_id, input_mask, segment_id, label_id in zip(input_ids, input_masks, segment_ids, label_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [0] * (self._sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self._sequence_length - len(segment_id)))

                pad_label_ids.append(label_id + [label_to_index["o"]] * (self._sequence_length - len(label_id)))

                sequence_len.append(len(input_id))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

                pad_label_ids.append(label_id[:self._sequence_length])

                sequence_len.append(self._sequence_length)

        return pad_input_ids, pad_input_masks, pad_segment_ids, pad_label_ids, sequence_len

    def gen_data(self, file_path, is_training=True):
        """
        生成数据
        :param file_path:
        :param is_training:
        :return:
        """

        # 1，读取原始数据
        inputs, labels = self.read_data(file_path)
        print("read finished")

        if is_training:
            uni_label = list(set(chain(*labels)))
            label_to_index = dict(zip(uni_label, list(range(len(uni_label)))))
            with open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf8") as fw:
                json.dump(label_to_index, fw, indent=0, ensure_ascii=False)
        else:
            with open(os.path.join(self.__output_path, "label_to_index.json"), "r", encoding="utf8") as fr:
                label_to_index = json.load(fr)

        # 2，输入转索引
        inputs_ids, input_masks, segment_ids, labels = self.trans_to_index(inputs, labels)
        print("index transform finished")

        # 3，标签转索引
        labels_ids = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        # 4, padding
        inputs_ids, input_masks, segment_ids, labels_ids, sequence_len = self.padding(inputs_ids,
                                                                                      input_masks,
                                                                                      segment_ids,
                                                                                      labels_ids,
                                                                                      label_to_index)

        for i in range(5):
            print("line {}: *****************************************".format(i))
            print("input: ", inputs[i])
            print("input_id: ", inputs_ids[i])
            print("input_mask: ", input_masks[i])
            print("segment_id: ", segment_ids[i])
            print("label_id: ", labels_ids[i])

        return inputs_ids, input_masks, segment_ids, labels_ids, sequence_len, label_to_index

    def next_batch(self, input_ids, input_masks, segment_ids, label_ids, sequence_len):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :param sequence_len:
        :return:
        """
        z = list(zip(input_ids, input_masks, segment_ids, label_ids, sequence_len))
        random.shuffle(z)
        input_ids, input_masks, segment_ids, label_ids, sequence_len = zip(*z)

        num_batches = len(input_ids) // self._batch_size

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
            batch_input_ids = input_ids[start: end]
            batch_input_masks = input_masks[start: end]
            batch_segment_ids = segment_ids[start: end]
            batch_label_ids = label_ids[start: end]
            batch_sequence_len = sequence_len[start: end]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       label_ids=batch_label_ids,
                       sequence_len=batch_sequence_len)
