
"""
定义性能指标函数
"""


def mean(item):
    return sum(item) / len(item)


def get_chunk_type(index, index_to_label):
    """
    对实体的标签进行分割，返回实体的位置和实体的名称
    """
    label_name = index_to_label[index]
    label_class, label_type = label_name.split("-")

    return label_name, label_class, label_type


def get_chunk(sequence, label_to_index):
    """
    给定一个标注序列，将实体和位置组合起来，放置在一个列表中
    """
    unentry = [label_to_index["o"]]
    index_to_label = {index: label for label, index in label_to_index.items()}
    chunks = []
    chunk_type, chunk_start = None, None
    for index, label in enumerate(sequence):
        if label in unentry:
            # 如果非实体词
            if chunk_type is None:
                # 若chunk_type为None，表明上一个词是非实体，继续跳过
                continue
            else:
                # 若chunkType非None，则上面的是一个实体，而当前非实体，则将上一个实体chunk加入到chunks中
                # 主要为序列中的这种情况，O,B-PER,I-PER,O 这也是最常见的情况
                chunk = (chunk_type, chunk_start, index-1)
                chunks.append(chunk)
                chunk_type, chunk_start = None, None

        if label not in unentry:
            # 如果是实体词，在这里的label是索引表示的label
            label_name, label_chunk_class, label_chunk_type = get_chunk_type(label, index_to_label)
            if chunk_type is None:
                # 若当前chunk_type为None，则表明上一个词是非实体词
                chunk_type, chunk_start = label_chunk_type, index
            elif label_chunk_type == chunk_type:
                # 若实体类型和上一个相同，则做如下判断
                if index == (len(sequence) - 1):
                    # 若当前词是序列中的最后一个词，则直接返回chunk
                    chunk = (chunk_type, chunk_start, index)
                    chunks.append(chunk)

                # 若出现两个相同的实体连在一块，则做如下操作
                elif label_chunk_class == "B":
                    chunk = (chunk_type, chunk_start, index - 1)
                    chunks.append(chunk)
                    chunk_type, chunk_start = label_chunk_type, index
                else:
                    # 若当前非最后一个词，则跳过
                    continue
            elif label_chunk_type != chunk_type:
                # 若当前词和上一个词类型不同，则将上一个实体chunk加入到chunks中，接着继续下一个chunk
                # 主要体现在两个实体相连的序列中，如B-PER,I-PER,B-LOC,I-LOC
                chunk = (chunk_type, chunk_start, index-1)
                chunks.append(chunk)
                chunk_type, chunk_start = label_chunk_type, index

    return chunks


def gen_metrics(true_y, pred_y, label_to_index):
    """
    生成f1值，recall, precision
    precision = 识别的正确实体数/识别出的实体数
    recall = 识别的正确实体数/样本的实体数
    """
    correct_preds = 0  # 识别出的正确实体数
    all_preds = 0  # 识别出的实体数
    all_trues = 0  # 样本的真实实体数

    true_chunks = get_chunk(true_y.tolist(), label_to_index)
    pred_chunks = get_chunk(pred_y.tolist(), label_to_index)
    correct_preds += len(set(true_chunks) & set(pred_chunks))
    all_preds += len(pred_chunks)
    all_trues += len(true_chunks)

    precision = correct_preds / all_preds if correct_preds > 0 else 0
    recall = correct_preds / all_trues if correct_preds > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if correct_preds > 0 else 0

    return round(f1, 4), round(precision, 4), round(recall, 4)