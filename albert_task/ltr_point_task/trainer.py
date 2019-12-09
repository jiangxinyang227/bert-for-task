import os
import json
import argparse
import sys

sys.path.append(os.path.dirname(os.getcwd()))
import tensorflow as tf
from albert import modeling
from model import ALBertPointLTR
from data_helper import TrainData
from metrics import mean, get_multi_metrics


class Trainer(object):
    def __init__(self, args):
        self.args = args
        with open(args.config_path, "r") as fr:
            self.config = json.load(fr)
        self.__bert_checkpoint_path = os.path.join(self.config["bert_model_path"], "albert_model.ckpt")

        # 加载数据集
        self.data_obj = self.load_data()
        self.t_in_ids_a, self.t_in_masks_a, self.t_seg_ids_a, self.t_in_ids_b, self.t_in_masks_b, self.t_seg_ids_b, \
        self.t_lab_ids, lab_to_idx = self.data_obj.gen_data(self.config["train_data"])

        self.e_in_ids_a, self.e_in_masks_a, self.e_seg_ids_a, self.e_in_ids_b, self.e_in_masks_b, self.e_seg_ids_b, \
        self.e_lab_ids, lab_to_idx = self.data_obj.gen_data(self.config["eval_data"], is_training=False)

        print("train data size: {}".format(len(self.t_lab_ids)))
        print("eval data size: {}".format(len(self.e_lab_ids)))
        self.label_list = [value for key, value in lab_to_idx.items()]
        print("label numbers: ", len(self.label_list))

        num_train_steps = int(
            len(self.t_lab_ids) / self.config["batch_size"] * self.config["epochs"])
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
        model = ALBertPointLTR(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step)
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

            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))

                for batch in self.data_obj.next_batch(self.t_in_ids_a, self.t_in_masks_a, self.t_seg_ids_a,
                                                      self.t_in_ids_b, self.t_in_masks_b, self.t_seg_ids_b,
                                                      self.t_lab_ids):
                    loss, predictions = self.model.train(sess, batch)

                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch["label_ids"],
                                                                  labels=self.label_list)
                    print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        current_step, loss, acc, recall, prec, f_beta))

                    current_step += 1
                    if self.data_obj and current_step % self.config["checkpoint_every"] == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.data_obj.next_batch(self.e_in_ids_a, self.e_in_masks_a, self.e_seg_ids_a,
                                                                   self.e_in_ids_b, self.e_in_masks_b, self.e_seg_ids_b,
                                                                   self.e_lab_ids):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)

                            acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                          true_y=eval_batch["label_ids"],
                                                                          labels=self.label_list)
                            eval_accs.append(acc)
                            eval_recalls.append(recall)
                            eval_precs.append(prec)
                            eval_f_betas.append(f_beta)
                        print("\n")
                        print("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                        print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="config path of model")
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.train()
