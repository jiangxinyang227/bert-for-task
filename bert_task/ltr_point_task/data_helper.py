
import os
import json
import random
import sys
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
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        text_as = []
        text_bs = []
        labels = []
        with open(file_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    text_a, text_b, label = line.strip().split("\t")
                    text_as.append(text_a)
                    text_bs.append(text_b)
                    labels.append(label)
                except:
                    continue

        return text_as, text_bs, labels

    def trans_to_index(self, texts):
        """
        将输入转化为索引表示
        :param texts: 输入
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.__vocab_path, do_lower_case=True)
        input_ids = []
        input_masks = []
        segment_ids = []

        for text in texts:
            text = tokenization.convert_to_unicode(text)
            tokens = tokenizer.tokenize(text)
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

            input_id = tokenizer.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)
            input_masks.append([1] * len(input_id))
            segment_ids.append([0] * len(input_id))

        return input_ids, input_masks, segment_ids

    @staticmethod
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, input_ids, input_masks, segment_ids):
        """
        对序列进行补全
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :return:
        """
        pad_input_ids, pad_input_masks, pad_segment_ids = [], [], []
        for input_id, input_mask, segment_id in zip(input_ids, input_masks, segment_ids):
            if len(input_id) < self._sequence_length:
                pad_input_ids.append(input_id + [0] * (self._sequence_length - len(input_id)))
                pad_input_masks.append(input_mask + [0] * (self._sequence_length - len(input_mask)))
                pad_segment_ids.append(segment_id + [0] * (self._sequence_length - len(segment_id)))
            else:
                pad_input_ids.append(input_id[:self._sequence_length])
                pad_input_masks.append(input_mask[:self._sequence_length])
                pad_segment_ids.append(segment_id[:self._sequence_length])

        return pad_input_ids, pad_input_masks, pad_segment_ids

    def gen_data(self, file_path, is_training=True):
        """
        生成数据
        :param file_path:
        :param is_training
        :return:
        """

        # 1，读取原始数据
        text_as, text_bs, labels = self.read_data(file_path)
        print("read finished")

        if is_training:
            uni_label = list(set(labels))
            label_to_index = dict(zip(uni_label, list(range(len(uni_label)))))
            with open(os.path.join(self.__output_path, "label_to_index.json"), "w", encoding="utf8") as fw:
                json.dump(label_to_index, fw, indent=0, ensure_ascii=False)
        else:
            with open(os.path.join(self.__output_path, "label_to_index.json"), "r", encoding="utf8") as fr:
                label_to_index = json.load(fr)

        # 2，输入转索引
        input_ids_a, input_masks_a, segment_ids_a = self.trans_to_index(text_as)
        input_ids_b, input_masks_b, segment_ids_b = self.trans_to_index(text_bs)
        print("index transform finished")

        input_ids_a, input_masks_a, segment_ids_a = self.padding(input_ids_a, input_masks_a, segment_ids_a)
        input_ids_b, input_masks_b, segment_ids_b = self.padding(input_ids_b, input_masks_b, segment_ids_b)

        # 3，标签转索引
        label_ids = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        for i in range(5):
            print("line {}: *****************************************".format(i))
            print("text_a: ", text_as[i])
            print("text_b: ", text_bs[i])
            print("input_id_a: ", input_ids_a[i])
            print("input_mask_a: ", input_masks_a[i])
            print("segment_id_a: ", segment_ids_a[i])
            print("input_id_b: ", input_ids_b[i])
            print("input_mask_b: ", input_masks_b[i])
            print("segment_id_b: ", segment_ids_b[i])
            print("label_id: ", labels[i])

        return input_ids_a, input_masks_a, segment_ids_a, input_ids_b, input_masks_b, segment_ids_b, label_ids, label_to_index

    def next_batch(self, input_ids_a, input_masks_a, segment_ids_a, input_ids_b,
                   input_masks_b, segment_ids_b, label_ids):
        """
        生成batch数据
        :param input_ids_a:
        :param input_masks_a:
        :param segment_ids_a:
        :param input_ids_b:
        :param input_masks_b:
        :param segment_ids_b:
        :param label_ids:
        :return:
        """
        z = list(zip(input_ids_a, input_masks_a, segment_ids_a, input_ids_b, input_masks_b, segment_ids_b, label_ids))
        random.shuffle(z)
        input_ids_a, input_masks_a, segment_ids_a, input_ids_b, input_masks_b, segment_ids_b, label_ids = zip(*z)

        num_batches = len(input_ids_a) // self._batch_size

        for i in range(num_batches):
            start = i * self._batch_size
            end = start + self._batch_size
            batch_input_ids_a = input_ids_a[start: end]
            batch_input_masks_a = input_masks_a[start: end]
            batch_segment_ids_a = segment_ids_a[start: end]

            batch_input_ids_b = input_ids_b[start: end]
            batch_input_masks_b = input_masks_b[start: end]
            batch_segment_ids_b = segment_ids_b[start: end]

            batch_label_ids = label_ids[start: end]

            yield dict(input_ids_a=batch_input_ids_a,
                       input_masks_a=batch_input_masks_a,
                       segment_ids_a=batch_segment_ids_a,
                       input_ids_b=batch_input_ids_b,
                       input_masks_b=batch_input_masks_b,
                       segment_ids_b=batch_segment_ids_b,
                       label_ids=batch_label_ids)

