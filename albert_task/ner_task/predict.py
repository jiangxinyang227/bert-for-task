import json
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

import tensorflow as tf
from model import ALBertNer
from albert import tokenization
from metrics import get_chunk


class Predictor(object):
    def __init__(self, config):
        self.model = None
        self.config = config

        self.output_path = config["output_path"]
        self.vocab_path = os.path.join(config["bert_model_path"], "vocab.txt")
        self.label_to_index = self.load_vocab()
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来

        with open(os.path.join(self.output_path, "label_to_index.json"), "r") as f:
            label_to_index = json.load(f)

        return label_to_index

    def padding(self, input_id, input_mask, segment_id):
        """
        对序列进行补全
        :param input_id:
        :param input_mask:
        :param segment_id:
        :return:
        """

        if len(input_id) < self.sequence_length:
            pad_input_id = input_id + [0] * (self.sequence_length - len(input_id))
            pad_input_mask = input_mask + [0] * (self.sequence_length - len(input_mask))
            pad_segment_id = segment_id + [0] * (self.sequence_length - len(segment_id))
            sequence_len = len(input_id)
        else:
            pad_input_id = input_id[:self.sequence_length]
            pad_input_mask = input_mask[:self.sequence_length]
            pad_segment_id = segment_id[:self.sequence_length]
            sequence_len = self.sequence_length

        return pad_input_id, pad_input_mask, pad_segment_id, sequence_len

    def sentence_to_idx(self, text):
        """
        将分词后的句子转换成idx表示
        :return:
        """
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=True)

        tokens = []
        for token in text:
            token = tokenizer.tokenize(token)
            tokens.extend(token)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_id = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_id)
        segment_id = [0] * len(input_id)

        input_id, input_mask, segment_id, sequence_len = self.padding(input_id, input_mask, segment_id)

        return [input_id], [input_mask], [segment_id], [sequence_len]

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
        self.model = ALBertNer(config=self.config, is_training=False)

    def predict(self, text):
        """
        给定分词后的句子，预测其分类结果
        :param text:
        :return:
        """
        input_ids, input_masks, segment_ids, sequence_len = self.sentence_to_idx(text)

        prediction = self.model.infer(self.sess,
                                      dict(input_ids=input_ids,
                                           input_masks=input_masks,
                                           segment_ids=segment_ids,
                                           sequence_len=sequence_len)).tolist()
        print(prediction)
        chunks = get_chunk(prediction, self.label_to_index)
        return chunks


