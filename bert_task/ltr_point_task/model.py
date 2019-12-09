import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf

from bert import modeling
from bert import optimization


class BertPointLTR(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__learning_rate = config["learning_rate"]
        self.__neg_threshold = config["neg_threshold"]
        self.__batch_size = config["batch_size"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.input_ids_a = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids_a')
        self.input_masks_a = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask_a')
        self.segment_ids_a = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids_a')

        self.input_ids_b = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids_b')
        self.input_masks_b = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask_b')
        self.segment_ids_b = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids_b')

        self.label_ids = tf.placeholder(dtype=tf.int32, shape=[None], name="label_ids")

        self.concat_input_ids = tf.concat([self.input_ids_a, self.input_ids_b], axis=0, name="concat_input_ids")
        self.concat_input_masks = tf.concat([self.input_masks_a, self.input_masks_b], axis=0, name="concat_input_masks")
        self.concat_segment_ids = tf.concat([self.segment_ids_a, self.segment_ids_b], axis=0, name="concat_segment_ids")

        self.built_model()
        self.init_saver()

    def built_model(self):
        bert_config = modeling.BertConfig.from_json_file(self.__bert_config_path)

        model = modeling.BertModel(config=bert_config,
                                   is_training=self.__is_training,
                                   input_ids=self.concat_input_ids,
                                   input_mask=self.concat_input_masks,
                                   token_type_ids=self.concat_segment_ids,
                                   use_one_hot_embeddings=False)
        concat_output = model.get_pooled_output()

        output_a, output_b = tf.split(concat_output, [self.__batch_size] * 2, axis=0)

        # -------------------------------------------------------------------------------------------
        # 余弦相似度 + 对比损失
        # -------------------------------------------------------------------------------------------
        with tf.name_scope("cosine_similarity"):
            # [batch_size]
            norm_a = tf.sqrt(tf.reduce_sum(tf.square(output_a), axis=-1))
            # [batch_size]
            norm_b = tf.sqrt(tf.reduce_sum(tf.square(output_b), axis=-1))
            # [batch_size]
            dot = tf.reduce_sum(tf.multiply(output_a, output_b), axis=-1)
            # [batch_size]
            norm = norm_a * norm_b
            # [batch_size]
            self.similarity = tf.div(dot, norm, name="similarity")
            self.predictions = tf.cast(tf.greater_equal(self.similarity, self.__neg_threshold), tf.int32,
                                       name="predictions")

        with tf.name_scope("loss"):
            # 预测为正例的概率
            pred_pos_prob = tf.square((1 - self.similarity))
            cond = (self.similarity > self.__neg_threshold)
            zeros = tf.zeros_like(self.similarity, dtype=tf.float32)
            pred_neg_prob = tf.where(cond, tf.square(self.similarity), zeros)
            self.label_ids = tf.cast(self.label_ids, dtype=tf.float32)
            losses = self.label_ids * pred_pos_prob + (1 - self.label_ids) * pred_neg_prob
            self.loss = tf.reduce_mean(losses, name="loss")

        # --------------------------------------------------------------------------------------------
        # # 曼哈顿距离 + 二元交叉熵
        # --------------------------------------------------------------------------------------------
        # with tf.name_scope("manhattan_distance"):
        #     man_distance = tf.reduce_sum(tf.abs(output_a - output_b), -1)
        #     self.similarity = tf.exp(-man_distance)
        #     self.predictions = tf.cast(tf.greater_equal(self.similarity, 0.5), tf.int32, name="predictions")
        #
        # with tf.name_scope("loss"):
        #     losses = self.label_ids * tf.log(self.similarity) + (1 - self.label_ids) * tf.log(1 - self.similarity)
        #     self.loss = tf.reduce_mean(-losses, name="loss")

        with tf.name_scope('train_op'):
            self.train_op = optimization.create_optimizer(
                self.loss, self.__learning_rate, self.__num_train_step, self.__num_warmup_step, use_tpu=False)

    def init_saver(self):
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """

        feed_dict = {self.input_ids_a: batch["input_ids_a"],
                     self.input_masks_a: batch["input_masks_a"],
                     self.segment_ids_a: batch["segment_ids_a"],
                     self.input_ids_b: batch["input_ids_b"],
                     self.input_masks_b: batch["input_masks_b"],
                     self.segment_ids_b: batch["segment_ids_b"],
                     self.label_ids: batch["label_ids"]}

        # 训练模型
        _, loss, predictions = sess.run([self.train_op, self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.input_ids_a: batch["input_ids_a"],
                     self.input_masks_a: batch["input_masks_a"],
                     self.segment_ids_a: batch["segment_ids_a"],
                     self.input_ids_b: batch["input_ids_b"],
                     self.input_masks_b: batch["input_masks_b"],
                     self.segment_ids_b: batch["segment_ids_b"],
                     self.label_ids: batch["label_ids"]}

        loss, predictions = sess.run([self.loss, self.predictions], feed_dict=feed_dict)
        return loss, predictions

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {self.input_ids_a: batch["input_ids_a"],
                     self.input_masks_a: batch["input_masks_a"],
                     self.segment_ids_a: batch["segment_ids_a"],
                     self.input_ids_b: batch["input_ids_b"],
                     self.input_masks_b: batch["input_masks_b"],
                     self.segment_ids_b: batch["segment_ids_b"]}

        predict = sess.run(self.predictions, feed_dict=feed_dict)

        return predict
