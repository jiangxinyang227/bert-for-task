import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf

from bert import modeling
from bert import optimization


class BertPairLTR(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")
        self.__num_samples = config["num_samples"]
        self.__learning_rate = config["learning_rate"]
        self.__margin = config["margin"]
        self.__batch_size = config["batch_size"]
        self.__sequence_length = config["sequence_length"]
        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.input_ids_a = tf.placeholder(dtype=tf.int32,
                                          shape=[self.__batch_size, self.__sequence_length],
                                          name='input_ids_a')
        self.input_masks_a = tf.placeholder(dtype=tf.int32,
                                            shape=[self.__batch_size, self.__sequence_length],
                                            name='input_mask_a')
        self.segment_ids_a = tf.placeholder(dtype=tf.int32,
                                            shape=[self.__batch_size, self.__sequence_length],
                                            name='segment_ids_a')

        self.input_ids_b = tf.placeholder(dtype=tf.int32,
                                          shape=[self.__batch_size * self.__num_samples, self.__sequence_length],
                                          name='input_ids_b')
        self.input_masks_b = tf.placeholder(dtype=tf.int32,
                                            shape=[self.__batch_size * self.__num_samples, self.__sequence_length],
                                            name='input_mask_b')
        self.segment_ids_b = tf.placeholder(dtype=tf.int32,
                                            shape=[self.__batch_size * self.__num_samples, self.__sequence_length],
                                            name='segment_ids_b')

        # [batch_size*(num_samples + 1), sequence_length]
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

        output_a, output_b = tf.split(concat_output, [self.__batch_size, self.__batch_size * self.__num_samples],
                                      axis=0)

        with tf.name_scope("reshape_output_b"):
            # batch_size 个tensor：[neg_samples, hidden_size]
            split_output_b = tf.split(output_b, [self.__num_samples] * self.__batch_size, axis=0)
            # batch_size 个tensor: [1, neg_samples, hidden_size]
            expand_output_b = [tf.expand_dims(tensor, 0) for tensor in split_output_b]
            # [batch_size, num_samples, hidden_size]
            reshape_output_b = tf.concat(expand_output_b, axis=0)

        with tf.name_scope("cosine_similarity"):
            # [batch_size, 1, hidden_size]
            expand_output_a = tf.expand_dims(output_a, 1)
            # [batch_size, 1]
            norm_a = tf.sqrt(tf.reduce_sum(tf.square(expand_output_a), -1))
            # [batch_size, n_samples]
            norm_b = tf.sqrt(tf.reduce_sum(tf.square(reshape_output_b), -1))
            # [batch_size, n_samples]
            dot = tf.reduce_sum(tf.multiply(expand_output_a, reshape_output_b), axis=-1)

            # [batch_size, n_samples]
            norm = norm_a * norm_b

            self.similarity = tf.div(dot, norm, name="similarity")
            self.predictions = tf.argmax(self.similarity, -1, name="predictions")

        with tf.name_scope("loss"):
            if self.__num_samples == 2:
                pos_similarity = tf.reshape(tf.slice(self.similarity, [0, 0], [self.__batch_size, 1]),
                                            [self.__batch_size])
                neg_similarity = tf.reshape(tf.slice(self.similarity,
                                                     [0, 1],
                                                     [self.__batch_size, self.__num_samples - 1]),
                                            [self.__batch_size])
                distance = self.__margin - pos_similarity + neg_similarity
                zeros = tf.zeros_like(distance, dtype=tf.float32)
                cond = (distance >= zeros)
                losses = tf.where(cond, distance, zeros)
                self.loss = tf.reduce_mean(losses, name="loss")
            else:
                pos_similarity = tf.exp(tf.reshape(tf.slice(self.similarity, [0, 0], [self.__batch_size, 1]),
                                                   [self.__batch_size]))
                neg_similarity = tf.exp(
                    tf.slice(self.similarity, [0, 1], [self.__batch_size, self.__num_samples - 1])
                )
                norm_seg_similarity = tf.reduce_sum(neg_similarity, axis=-1)
                pos_prob = tf.div(pos_similarity, norm_seg_similarity)
                self.loss = tf.reduce_mean(-tf.log(pos_prob), name="loss")

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
                     self.segment_ids_b: batch["segment_ids_b"]}

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
                     self.segment_ids_b: batch["segment_ids_b"]}

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
