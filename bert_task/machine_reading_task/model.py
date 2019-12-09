import os
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf

from bert import modeling
from bert import optimization


class BertMachineReading(object):
    def __init__(self, config, is_training=True, num_train_step=None, num_warmup_step=None):
        self.__bert_config_path = os.path.join(config["bert_model_path"], "bert_config.json")

        self.__is_training = is_training
        self.__num_train_step = num_train_step
        self.__num_warmup_step = num_warmup_step

        self.__max_length = config["max_length"]
        self.__learning_rate = config["learning_rate"]

        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.__max_length], name='input_ids')
        self.input_masks = tf.placeholder(dtype=tf.int32, shape=[None, self.__max_length], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, self.__max_length], name='segment_ids')
        self.start_position = tf.placeholder(dtype=tf.int32, shape=[None], name="start_position")
        self.end_position = tf.placeholder(dtype=tf.int32, shape=[None], name="end_position")

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

        final_hidden = model.get_sequence_output()

        final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
        seq_length = final_hidden_shape[1]
        hidden_size = final_hidden_shape[2]

        with tf.name_scope("output"):
            output_weights = tf.get_variable(
                "output_weights", [2, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable(
                "output_bias", [2], initializer=tf.zeros_initializer())

            final_hidden_matrix = tf.reshape(final_hidden,
                                             [-1, hidden_size])
            logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            logits = tf.reshape(logits, [-1, seq_length, 2])
            logits = tf.transpose(logits, [2, 0, 1])

            unstacked_logits = tf.unstack(logits, axis=0)

            # [batch_size, seq_length]
            start_logits, end_logits = (unstacked_logits[0], unstacked_logits[1])

            self.start_logits = start_logits
            self.end_logits = end_logits

        if self.__is_training:
            with tf.name_scope("loss"):
                start_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_logits,
                                                                              labels=self.start_position)
                end_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_logits,
                                                                            labels=self.end_position)

                losses = tf.concat([start_losses, end_losses], axis=0)
                self.loss = tf.reduce_mean(losses, name="loss")

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

        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"],
                     self.start_position: batch["start_position"],
                     self.end_position: batch["end_position"]}

        # 训练模型
        _, loss, start_logits, end_logits = sess.run([self.train_op, self.loss, self.start_logits, self.end_logits],
                                                     feed_dict=feed_dict)
        return loss, start_logits, end_logits

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
                     self.start_position: batch["start_position"],
                     self.end_position: batch["end_position"]}

        start_logits, end_logits = sess.run([self.start_logits, self.end_logits], feed_dict=feed_dict)
        return start_logits, end_logits

    def infer(self, sess, batch):
        """
        预测新数据
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 预测结果
        """
        feed_dict = {self.input_ids: batch["input_ids"],
                     self.input_masks: batch["input_masks"],
                     self.segment_ids: batch["segment_ids"]}

        start_logits, end_logits = sess.run([self.start_logits, self.end_logits], feed_dict=feed_dict)

        return start_logits, end_logits
