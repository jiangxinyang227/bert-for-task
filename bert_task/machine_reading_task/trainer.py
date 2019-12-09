import os
import json
import argparse
import time
import collections
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from bert import modeling
from model import BertMachineReading
from data_helper import TrainData
from metrics import get_eval, write_predictions


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "bert_model.ckpt")

        # 加载数据集
        self.data_obj = self.load_data()
        self.t_features = self.data_obj.gen_data(self.config["train_data"])

        self.e_examples, self.e_features = self.data_obj.gen_data(self.config["eval_data"], is_training=False)
        print("train data size: {}".format(len(self.t_features)))
        print("eval data size: {}".format(len(self.e_features)))

        num_train_steps = int(
            len(self.t_features) / self.config["batch_size"] * self.config["epochs"])
        num_warmup_steps = int(num_train_steps * self.config["warmup_rate"])
        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        data_obj = TrainData(self.config)
        return data_obj

    def create_model(self, num_train_step, num_warmup_step):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        model = BertMachineReading(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step)
        return model

    def train(self):
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.__bert_checkpoint_path)
            print("init bert model params")
            tf.train.init_from_checkpoint(self.__bert_checkpoint_path, assignment_map)
            print("init bert model params done")
            sess.run(tf.variables_initializer(tf.global_variables()))

            current_step = 0
            start = time.time()
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.t_features):
                    loss, start_logits, end_logits = self.model.train(sess, batch)
                    # print("start: ", start_logits)
                    # print("end: ", end_logits)
                    print("train: step: {}, loss: {}".format(current_step, loss))

                    current_step += 1
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        all_results = []
                        for eval_batch in self.data_obj.next_batch(self.e_features, is_training=False):
                            start_logits, end_logits = self.model.eval(sess, eval_batch)

                            for unique_id, start_logit, end_logit in zip(eval_batch["unique_id"],
                                                                         start_logits,
                                                                         end_logits):
                                all_results.append(dict(unique_id=unique_id,
                                                        start_logits=start_logit.tolist(),
                                                        end_logits=end_logit.tolist()))

                        with open("output/cmrc2018/results.json", "w", encoding="utf8") as fw:
                            json.dump(all_results, fw, indent=4, ensure_ascii=False)

                        write_predictions(all_examples=self.e_examples,
                                          all_features=self.e_features,
                                          all_results=all_results,
                                          n_best_size=self.config["n_best_size"],
                                          max_answer_length=self.config["max_answer_length"],
                                          output_prediction_file=self.config["output_predictions_path"],
                                          output_nbest_file=self.config["output_nbest_path"])

                        result = get_eval(original_file=self.config["eval_data"],
                                          prediction_file=self.config["output_predictions_path"])

                        print("\n")
                        print("eval:  step: {}, f1: {}, em: {}".format(current_step, result["f1"], result["em"]))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)

            end = time.time()
            print("total train time: ", end - start)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
