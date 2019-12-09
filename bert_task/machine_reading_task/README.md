#### config文件解读

##### 以cmrc_config.json为例

* model_name：模型名称
* epochs：迭代epoch的数量
* checkpoint_every：间隔多少步保存一次模型
* eval_every：间隔多少步验证一次模型
* learning_rate：学习速率，推荐2e-5， 5e-5， 1e-4
* max_length：输入到模型中的最大长度，建议设置为512
* doc_stride：对于context长度较长的时候会分成多个doc，采用滑动窗口的形式分doc，这个是滑动窗口的大小，建议设为128
* query_length：输入的问题的最大长度
* max_answer_length：生成的回答的最大长度
* n_best_size：获取分数最高的前n个
* batch_size：单GPU时不要超过32
* num_classes：文本分类的类别数量
* warmup_rate：训练时的预热比例，建议0.05， 0.1
* output_path：输出文件夹，用来存储label_to_index等文件
* output_predictions_path：训练时在验证集上预测的最佳结果保存路径
* output_nbest_path：训练时在验证集上预测的n个最佳结果的保存路径
* bert_model_path：预训练模型文件夹路径
* train_data：训练数据路径
* eval_data：验证数据路径
* ckpt_model_path：checkpoint模型文件保存路径