import tensorflow as tf
import numpy as np
from bert_task.bert import modeling
from bert_task.bert import tokenization
from bert_task.bert import optimization

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_batch_size', 6, 'define the train batch size')
flags.DEFINE_integer('num_train_epochs', 3, 'define the num train epochs')
flags.DEFINE_float('warmup_proportion', 0.1, 'define the warmup proportion')
flags.DEFINE_float('learning_rate', 5e-5, 'the initial learning rate for adam')
flags.DEFINE_bool('is_training', True, 'define weather fine-tune the bert model')
flags.DEFINE_integer('max_sentence_len', 512, 'define the max len of sentence')
flags.DEFINE_bool('task_train', True, 'define the train task')
flags.DEFINE_bool('task_predict', True, 'define the predict task')


def get_start_end_index(text, subtext):
    for i in range(len(text)):
        if text[i:i + len(subtext)] == subtext:
            return (i, i + len(subtext) - 1)
    return (-1, -1)


train_data = []
with open('data/train_data.txt', encoding='UTF-8') as fp:
    strLines = fp.readlines()
    strLines = [item.strip() for item in strLines]
    strLines = [eval(item) for item in strLines]
    train_data.extend(strLines)

test_data = []
with open('data/test_data.txt', encoding='UTF-8') as fp:
    strLines = fp.readlines()
    strLines = [item.strip() for item in strLines]
    strLines = [eval(item) for item in strLines]
    test_data.extend(strLines)

# config_path = r'D:\NLP_SOUNDAI\learnTensor\package9\bert\chinese_L-12_H-768_A-12\bert_config.json'
# checkpoint_path = r'D:\NLP_SOUNDAI\learnTensor\package9\bert\chinese_L-12_H-768_A-12\bert_model.ckpt'
# dict_path = r'D:\NLP_SOUNDAI\learnTensor\package9\bert\chinese_L-12_H-768_A-12\vocab.txt'
config_path = './bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = './bert/chinese_L-12_H-768_A-12/vocab.txt'
bert_config = modeling.BertConfig.from_json_file(config_path)
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)


def input_str_concat(inputList):
    assert len(inputList) == 2
    t, c = inputList
    newStr = '__%s__%s' % (c, t)
    newStr = newStr[:510]
    tokens = tokenizer.tokenize(newStr)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    return tokens, (input_ids, input_mask, segment_ids)


def sequence_padding(sequence):
    lenlist = [len(item) for item in sequence]
    maxlen = max(lenlist)
    return np.array([
        np.concatenate([item, [0] * (maxlen - len(item))]) if len(item) < maxlen else item for item in sequence
    ])


# get train data batch
def get_data_batch():
    batch_size = FLAGS.train_batch_size
    epoch = FLAGS.num_train_epochs
    for oneEpoch in range(epoch):
        num_batches = ((len(train_data) - 1) // batch_size) + 1
        for i in range(num_batches):
            batch_data = train_data[i * batch_size:(i + 1) * batch_size]
            yield_batch_data = {
                'input_ids': [],
                'input_mask': [],
                'segment_ids': [],
                'start_ids': [],
                'end_ids': []
            }
            for item in batch_data:
                tokens, (input_ids, input_mask, segment_ids) = input_str_concat(item[:-1])
                target_tokens = tokenizer.tokenize(item[2])

                start, end = get_start_end_index(tokens, target_tokens)

                start_ids = [0] * len(input_ids)
                end_ids = [0] * len(input_ids)
                start_ids[start] = 1
                end_ids[end] = 1
                yield_batch_data['input_ids'].append(input_ids)
                yield_batch_data['input_mask'].append(input_mask)
                yield_batch_data['segment_ids'].append(segment_ids)
                yield_batch_data['start_ids'].append(start_ids)
                yield_batch_data['end_ids'].append(end_ids)
            yield_batch_data['input_ids'] = sequence_padding(yield_batch_data['input_ids'])
            yield_batch_data['input_mask'] = sequence_padding(yield_batch_data['input_mask'])
            yield_batch_data['segment_ids'] = sequence_padding(yield_batch_data['segment_ids'])
            yield_batch_data['start_ids'] = sequence_padding(yield_batch_data['start_ids'])
            yield_batch_data['end_ids'] = sequence_padding(yield_batch_data['end_ids'])
            yield yield_batch_data


with tf.Graph().as_default(), tf.Session() as sess:
    input_ids_p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_ids_p')
    input_mask_p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='input_mask_p')
    segment_ids_p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='segment_ids_p')
    start_p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='start_p')
    end_p = tf.placeholder(dtype=tf.int64, shape=[None, None], name='end_p')

    model = modeling.BertModel(config=bert_config,
                               is_training=FLAGS.is_training,
                               input_ids=input_ids_p,
                               input_mask=input_mask_p,
                               token_type_ids=segment_ids_p,
                               use_one_hot_embeddings=False)
    output_layer = model.get_sequence_output()

    word_dim = output_layer.get_shape().as_list()[-1]
    output_reshape = tf.reshape(output_layer, shape=[-1, word_dim], name='output_reshape')

    with tf.variable_scope('weitht_and_bias', reuse=tf.AUTO_REUSE,
                           initializer=tf.truncated_normal_initializer(mean=0., stddev=0.05)):
        weight_start = tf.get_variable(name='weight_start', shape=[word_dim, 1])
        bias_start = tf.get_variable(name='bias_start', shape=[1])
        weight_end = tf.get_variable(name='weight_end', shape=[word_dim, 1])
        bias_end = tf.get_variable(name='bias_end', shape=[1])

    with tf.name_scope('predict_start_and_end'):
        pred_start = tf.einsum('ijk,kd->ijd', output_layer, weight_start)
        pred_start = tf.nn.bias_add(pred_start, bias_start)
        pred_start = tf.squeeze(pred_start, -1)

        pred_end = tf.einsum('ijk,kd->ijd', output_layer, weight_end)
        pred_end = tf.nn.bias_add(pred_end, bias_end)
        pred_end = tf.squeeze(pred_end, -1)

        pred_start_index = tf.argmax(pred_start, axis=1)
        pred_end_index = tf.argmax(pred_end, axis=1)

    with tf.name_scope('loss'):
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_start, labels=start_p))

        cumsum_start_p = 1 - tf.cumsum(start_p, axis=1)
        cumsum_start_p_10 = cumsum_start_p * 100000
        end_p -= cumsum_start_p_10

        loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred_end, labels=end_p))
        loss = loss1 + loss2

    with tf.name_scope('acc_predict'):
        start_acc_bool = tf.equal(tf.argmax(start_p, axis=1), tf.argmax(pred_start, axis=1))
        end_acc_bool = tf.equal(tf.argmax(end_p, axis=1), tf.argmax(pred_end, axis=1))
        start_acc = tf.reduce_mean(tf.cast(start_acc_bool, dtype=tf.float32))
        end_acc = tf.reduce_mean(tf.cast(end_acc_bool, dtype=tf.float32))
        total_acc = tf.reduce_mean(tf.cast(tf.reduce_all([start_acc_bool, end_acc_bool], axis=0), dtype=tf.float32))

    with tf.name_scope('train_op'):
        num_train_steps = int(
            len(train_data) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        train_op = optimization.create_optimizer(
            loss, FLAGS.learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

    tvars = tf.trainable_variables()
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, checkpoint_path)
    tf.train.init_from_checkpoint(checkpoint_path, assignment_map)
    sess.run(tf.variables_initializer(tf.global_variables()))

    if FLAGS.task_train:
        total_steps = 0
        for yield_batch_data in get_data_batch():
            total_steps += 1
            feed_dict = {
                input_ids_p: yield_batch_data['input_ids'],
                input_mask_p: yield_batch_data['input_mask'],
                segment_ids_p: yield_batch_data['segment_ids'],
                start_p: yield_batch_data['start_ids'],
                end_p: yield_batch_data['end_ids']
            }
            fetches = [train_op, loss, start_acc, end_acc, total_acc]

            _, loss_val, start_acc_val, end_acc_val, total_acc_val = sess.run(fetches, feed_dict=feed_dict)
            print('i : %s, loss : %s, start_acc : %s, end_acc : %s, total_acc : %s' % (
                total_steps, loss_val, start_acc_val, end_acc_val, total_acc_val))
        print('train task done ...')

    if FLAGS.task_predict:
        resultList = []
        for item in test_data:
            if item[1] == '其他':
                resultList.append('NaN')
            else:
                yield_batch_data = {
                    'input_ids': [],
                    'input_mask': [],
                    'segment_ids': [],
                }

                tokens, (input_ids, input_mask, segment_ids) = input_str_concat(item)

                yield_batch_data['input_ids'].append(input_ids)
                yield_batch_data['input_mask'].append(input_mask)
                yield_batch_data['segment_ids'].append(segment_ids)

                yield_batch_data['input_ids'] = sequence_padding(yield_batch_data['input_ids'])
                yield_batch_data['input_mask'] = sequence_padding(yield_batch_data['input_mask'])
                yield_batch_data['segment_ids'] = sequence_padding(yield_batch_data['segment_ids'])

                feed_dict = {
                    input_ids_p: yield_batch_data['input_ids'],
                    input_mask_p: yield_batch_data['input_mask'],
                    segment_ids_p: yield_batch_data['segment_ids']
                }

                fetches = [pred_start_index, pred_end_index]

                start_index, end_index = sess.run(fetches, feed_dict=feed_dict)
                start = start_index[0]
                end = end_index[0]
                oneResult = tokens[start:end + 1]

                if oneResult in item[0]:
                    resultList.append(oneResult)
                else:
                    if oneResult.upper() in item[0]:
                        resultList.append(oneResult.upper())
                    else:
                        originStr = item[0]
                        originStr = originStr.upper()
                        oneResult = oneResult.upper()

                        oneResult.replace('[UNK]', '').replace('#', '').strip()
                        resultList.append(oneResult)

        with open('result.txt', encoding='UTF-8', mode='a') as ff:
            ff.write('\n'.join(resultList) + '\n')





