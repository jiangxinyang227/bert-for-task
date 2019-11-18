import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf

from albert import modeling
from albert import optimization_finetuning as optimization
from bilstm_crf import BiLSTMCRF


class ALBertNer(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "albert_config.json")
        self.__num_classes = config["num_classes"]
        self.__learning_rate = config["learning_rate"]
        self.__ner_layers = config["ner_layers"]
        self.__ner_hidden_sizes = config["ner_hidden_sizes"]
        self.__max_len = config["sequence_length"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name="label_ids")
        self.sequence_len = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_len")
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=None, name="keep_prob")

        self.built_model()
        self.init_saver()

    def built_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        model = modeling.BertModel(config=bert_config,
                                   is_training=self.__is_training,
                                   input_ids=self.input_ids,
                                   input_mask=self.input_masks,
                                   token_type_ids=self.segment_ids,
                                   use_one_hot_embeddings=False)

        # 获取bert最后一层的输出
        output_layer = model.get_sequence_output()

        if self.__is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        ner_model = BiLSTMCRF(embedded_chars=output_layer,
                              hidden_sizes=self.__ner_hidden_sizes,
                              layers=self.__ner_layers,
                              keep_prob=self.keep_prob,
                              num_labels=self.__num_classes,
                              max_len=self.__max_len,
                              labels=self.label_ids,
                              sequence_lens=self.sequence_len,
                              is_training=self.__is_training)

        self.loss, self.true_y, self.predictions = ner_model.construct_graph()

        if self.__is_training:
            with tf.name_scope('train_op'):
                self.train_op = optimization.create_optimizer(
                    self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, dropout_rate):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :param dropout_rate: dropout rate
        :return: 损失和预测结果
        """

        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.keep_prob: dropout_rate}

        # 训练模型
        _, loss, true_y, predictions = sess.run([self.train_op, self.loss, self.true_y, self.predictions],
                                                feed_dict=feed_dict)
        return loss, true_y, predictions

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.label_ids: batch["label_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.keep_prob: 1.0}

        loss, true_y, predictions = sess.run([self.loss, self.true_y, self.predictions], feed_dict=feed_dict)
        return loss, true_y, predictions

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.sequence_len: batch["sequence_len"],
                     self.keep_prob: 1.0}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
