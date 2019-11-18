### BERT和ALBERT在下游任务中的应用
#### 本项目提供了易用的训练模式和预测模式，可以直接部署。也容易扩展到任何下游任务中

#### albert_task和bert_task文件夹中的内容基本一致
* albert_task/albert是albert的源码
* bert_task/bert是bert的源码
* 需要下载albert的预训练模型放置在albert_task下，bert的预训练模型放置在bert_task下
* 预训练模型的路径可以在xxx_config.json文件中配置

#### 目前提供了4大类的任务，classifier，sentence pair，ner，learning to rank(pair wise)。基准数据集来自chineseGLUE
* classifier包括tnews，inews，thucnews
* sentence pair包括bq，lcqmc，xnli
* ner包括msraner
* learning to rank(pair wise)是biendata上 **基于Adversarial Attack的问题等价性判别比赛**

#### 每个任务下的结构基本一致
* config：放置每个具体任务的配置文件，包括训练参数，数据路径，模型存储路径
* data_helper.py：数据预处理文件
* metrics.py：性能指标文件
* model.py：模型文件，可以很容易的实现bert和下游网络层的结合
* trainer.py：训练模型
* predict.py：预测代码，只需要实例化Predictor类，调用predict方法就可以预测

#### 训练数据格式
##### 文本分类数据格式
* title \<SEP> content \<SEP> label：有的数据中含有标题，有的只有正文，标题，正文，标签之间用\<SEP>符号分隔。
##### 句子对数据格式
* sentence A\<SEP>sentence B\<SEP>label：同样对于两个句子和标签采用\<SEP>符号分隔。
##### ner数据格式
###### 我们采用了BIO的格式标注，也可以采用BIOS, BIEO, BIEOS标注，将输入中的词和标注都用\t分隔。
* 慕 名 前 来 品 尝 玉 峰 茶 ， 领 略 茶 文 化 的 人 越 来 越 多 。\<SEP>o o o o o o B-ns I-ns o o o o o o o o o o o o o o

#### 训练模型
* 执行每个任务下的sh脚本即可，sh run.sh。只需要更改配置文件就可以训练不同的模型

#### 预测
* 执行albert_task中每个任务下的test.py文件就可以预测，bert_task同albert_task。