import argparse
import bert_score
import numpy as np
import tqdm
from sacrebleu import sentence_chrf
from table_utils import (
    extract_table_by_name,
    parse_text_to_table,
    is_empty_table,
)

bert_scorer = None
metric_cache = dict()  # cache some comparison operations

def parse_table_element_to_relation(table, i, j, row_header: bool, col_header: bool):
    """
    将表格中的元素解析为关系元组。

    参数:
        table (numpy.ndarray): 表格数据。
        i (int): 元素所在行的索引。
        j (int): 元素所在列的索引。
        row_header (bool): 是否包含行标题。
        col_header (bool): 是否包含列标题。

    返回:
        tuple: 解析后的关系元组。

    异常:
        AssertionError: 如果既不包含行标题也不包含列标题。
    """
    assert row_header or col_header
    relation = []
    if row_header:
        assert j > 0
        relation.append(table[i][0]) # 添加行标题
    if col_header:
        assert i > 0
        relation.append(table[0][j]) # 添加列标题
    relation.append(table[i][j]) # 添加数据
    return tuple(relation)


def parse_table_to_data(table, row_header: bool, col_header: bool):
    """
    将表格解析为数据。

    参数:
        table (numpy.ndarray): 表格数据。
        row_header (bool): 是否包含行标题。
        col_header (bool): 是否包含列标题。

    返回:
        tuple: 行标题集合，列标题集合，关系元组集合。

    异常:
        AssertionError: 如果既不包含行标题也不包含列标题。
    """
    if is_empty_table(table, row_header, col_header):
        return set(), set(), set()

    assert row_header or col_header
    row_headers = list(table[:, 0]) if row_header else []
    col_headers = list(table[0, :]) if col_header else []
    if row_header and col_headers: # 行列标题都有，左上角应为空
        row_headers = row_headers[1:]
        col_headers = col_headers[1:]

    row, col = table.shape
    relations = []
    for i in range(1 if col_header else 0, row): # 有列标题则从第一行开始读
        for j in range(1 if row_header else 0, col): # 有行标题则从第一列开始读
            if table[i][j] != "":
                relations.append(parse_table_element_to_relation(table, i, j, row_header, col_header))
    return set(row_headers), set(col_headers), set(relations) # 行标题，列标题，关系元组（行标题名，列标题名，数据）<若无行/列，则变成二维>


def parse_args():
    """
    解析命令行参数。

    返回:
        argparse.Namespace: 包含解析后参数的命名空间对象。

    异常:
        AssertionError: 如果既不包含行标题也不包含列标题。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp')
    parser.add_argument('tgt')
    parser.add_argument('--row-header', default=False, action="store_true")
    parser.add_argument('--col-header', default=False, action="store_true")
    parser.add_argument('--table-name', default=None)
    parser.add_argument('--metric', default='E', choices=['E', 'c', 'BS-scaled', ],
                        help="E: exact match\nc: chrf\nBS-scaled: re-scaled BERTScore")
    args = parser.parse_args()
    assert args.row_header or args.col_header
    print("Args", args)
    return args


def calc_similarity_matrix(tgt_data, pred_data, metric):
    """
    计算所有的目标数据和所有的预测数据之间的相似性矩阵，使用指定的度量标准。

    参数:
        tgt_data (list): 目标数据的列表。  （可能只是行标题，列标题，或者数据）
        pred_data (list): 预测数据的列表。
        metric (str): 用于计算相似性的度量标准。

    返回:
        numpy.ndarray: 相似性矩阵。

    异常:
        ValueError: 如果度量标准不受支持。
    """
    def calc_data_similarity(tgt, pred):
        """
        计算单个目标数据和预测数据之间的相似性。

        参数:
            tgt: 目标数据。
            pred: 预测数据。

        返回:
            相似性值。
        """
        if isinstance(tgt, tuple): # 如果tgt是元组，那么它是一个关系元组，它的第一个元素是行标题，第二个元素是列标题，第三个元素是数据。（若没有列，则是两个元素）
            ret = 1.0
            for tt, pp in zip(tgt, pred): # 也就是分别对行标题，列标题，数据计算相似度
                ret *= calc_data_similarity(tt, pp) # 对元组中的每个字符串计算相似度（递归计算，这时候因为不是元组了，所以就进入两个词之间的计算了）如果标题不相同，那么ret就会变成0；同样，如果数据不相同，ret也会变成0。
            return ret

        if (tgt, pred) in metric_cache: # 缓存
            return metric_cache[(tgt, pred)]

        if metric == 'E':
            # ret = int(tgt == pred) # 完全匹配，相等就是1，不等就是0
            # ret = int(tgt.strip().lower() == pred.strip().lower()) # zl优化，不区分大小写
            # 去除字符串首尾的空格，转换为小写，并去除开头的 "the "
            tgt_clean = tgt.strip().lower().lstrip("the ").lstrip()
            pred_clean = pred.strip().lower().lstrip("the ").lstrip()
            # 将所有的 "-" 替换为空格
            tgt_clean = tgt_clean.replace("-", " ")
            pred_clean = pred_clean.replace("-", " ")
            # 比较去除 "the " 后的字符串是否完全匹配
            ret = int(tgt_clean == pred_clean)
        elif metric == 'c':
            ret = sentence_chrf(pred, [tgt, ]).score / 100 # chrf分数
        elif metric == 'BS-scaled':
            global bert_scorer
            if bert_scorer is None:
                # 设置模型路径为本地路径
                model_type_local = "/workspace/wzl/datamining/text_to_table/wzl/roberta-large/"
                baseline_tsv_path = "/home/wangzilong/anaconda3/envs/py38_dm/lib/python3.8/site-packages/bert_score/rescale_baseline/en/roberta-large.tsv"

                bert_scorer = bert_score.BERTScorer(model_type=model_type_local, num_layers=8, lang="en", rescale_with_baseline=True, baseline_path=baseline_tsv_path)

                # bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True) # 这里无法下载模型，导致出错
            ret = bert_scorer.score([pred, ], [tgt, ])[2].item()
            ret = max(ret, 0)
            ret = min(ret, 1)
        else:
            raise ValueError(f"Metric cannot be {metric}")

        metric_cache[(tgt, pred)] = ret # 将计算结果存入缓存
        return ret

    return np.array([[calc_data_similarity(tgt, pred) for pred in pred_data] for tgt in tgt_data], dtype=np.float) # 计算相似度矩阵，tgt_data是目标数据的列表，pred_data是预测数据的列表，用每一个目标数据和每一个预测数据计算相似度，得到一个矩阵

def metrics_by_sim(tgt_data, pred_data, metric):
    """
    根据相似性矩阵计算指标。

    参数:
        tgt_data (list): 目标数据的列表。
        pred_data (list): 预测数据的列表。
        metric (str): 用于计算相似性的度量标准。

    返回:
        tuple: 精确率、召回率和 F1 值。

    异常:
        AssertionError: 如果既不包含行标题也不包含列标题。
    """
    sim = calc_similarity_matrix(tgt_data, pred_data, metric)  # (n_tgt, n_pred) matrix。 目标数据行数和预测数据列数的矩阵。
    prec = np.mean(np.max(sim, axis=0)) # n_pred个列（预测的个数）中最大的相似度的平均值。是通过取相似性矩阵sim沿着列方向（axis=0）的最大值，然后计算这些最大值的平均值。这意味着对于每一个预测数据，我们找到了与其最相似的目标数据，然后计算所有这些最相似值（精确度）的平均值。
    recall = np.mean(np.max(sim, axis=1)) # n_tgt个行（目标的个数）中最大的相似度的平均值。是通过取相似性矩阵sim沿着行方向（axis=1）的最大值，然后计算这些最大值的平均值。这意味着对于每一个目标数据，我们找到了与其最相似的预测数据，然后计算所有这些最相似值（召回率）的平均值。
    if recall != 1.0:
        print("Recall is not 1.0:, ", recall, sim, " 目标：", tgt_data, "预测：" ,pred_data)
    if prec + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * prec * recall / (prec + recall)
    return prec, recall, f1


if __name__ == "__main__":
    args = parse_args()
    # 它首先通过parse_args函数解析命令行参数，然后读取预测结果（hyp）和目标结果（tgt）的文件，并将每一行的内容解析为表格数据，存储在hyp_data和tgt_data列表中。
    hyp_data = []
    with open(args.hyp) as f:
        for line in f: # 逐行读取预测结果
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            hyp_data.append(parse_text_to_table(line)) # 将每一行的文字表格形式（模型的输出格式）的内容解析为numpy表格数据，并存储在hyp_data列表中
    tgt_data = []
    with open(args.tgt) as f:
        for line in f:
            line = line.strip()
            if args.table_name is not None: # 最终评测时没有传table_name
                line = extract_table_by_name(line, args.table_name)
            tgt_data.append(parse_text_to_table(line))
    # 接着，它初始化了一些空列表，用于存储行头、列头和关系的精确度、召回率和F1分数。
    row_header_precision = []
    row_header_recall = []
    row_header_f1 = []
    col_header_precision = []
    col_header_recall = []
    col_header_f1 = []
    relation_precision = []
    relation_recall = []
    relation_f1 = []
    # 然后，它遍历每一对预测结果和目标结果的表格数据。否则，它会解析表格数据，提取行头、列头和关系，然后计算这些数据的精确度、召回率和F1分数，并将这些分数添加到相应的列表中。
    for hyp_table, tgt_table in tqdm.tqdm(zip(hyp_data, tgt_data), total=len(hyp_data)): # 
        if is_empty_table(tgt_table, args.row_header, args.col_header):
            pass # 目标表格为空，基本不会出现这种情况
        elif hyp_table is None or is_empty_table(hyp_table, args.row_header, args.col_header): # 预测表格数据为空或者预测表格数据也为空，它会将精确度、召回率和F1分数都设置为0。
            if args.row_header:
                row_header_precision.append(0)
                row_header_recall.append(0)
                row_header_f1.append(0)
            if args.col_header:
                col_header_precision.append(0)
                col_header_recall.append(0)
                col_header_f1.append(0)
            relation_precision.append(0)
            relation_recall.append(0)
            relation_f1.append(0)
        else:
            # parse_table_to_data函数是用来将表格解析为数据的。它首先检查表格是否为空，如果为空，就返回空的集合。然后，它提取行头和列头，如果表格同时包含行头和列头，就将行头和列头的第一个元素去掉。最后，它遍历表格中的每一个元素，如果元素不为空，就将元素解析为关系，并添加到关系集合中。
            hyp_row_headers, hyp_col_headers, hyp_relations = parse_table_to_data(hyp_table, args.row_header,args.col_header) # 预测结果的行标题，列标题，和关系数据
            tgt_row_headers, tgt_col_headers, tgt_relations = parse_table_to_data(tgt_table, args.row_header,args.col_header)
            if args.row_header:
                p, r, f = metrics_by_sim(tgt_row_headers, hyp_row_headers, args.metric) # 比较预测的行标题和列标题的相似度。精确率、召回率和 F1 值。
                row_header_precision.append(p) # 将精确度添加到row_header_precision列表中
                row_header_recall.append(r)
                row_header_f1.append(f)
            if args.col_header:
                p, r, f = metrics_by_sim(tgt_col_headers, hyp_col_headers, args.metric)
                col_header_precision.append(p)
                col_header_recall.append(r)
                col_header_f1.append(f)
            if len(hyp_relations) == 0: # 没提取到行/列之间的数值
                relation_precision.append(0.0)
                relation_recall.append(0.0)
                relation_f1.append(0.0)
            else:
                p, r, f = metrics_by_sim(tgt_relations, hyp_relations, args.metric) # 比较预测的数据的相似度。精确率、召回率和 F1 值。
                relation_precision.append(p)
                relation_recall.append(r)
                relation_f1.append(f)
    # 如果args.row_header为真，代码会计算行头的精确度、召回率和F1分数，然后将结果打印出来。这些分数是通过对row_header_precision、row_header_recall和row_header_f1列表中的值求平均，然后乘以100得到的。
    if args.row_header:
        print("Row header: precision = %.2f; recall = %.2f; f1 = %.2f" % (
            np.mean(row_header_precision) * 100, np.mean(row_header_recall) * 100, np.mean(row_header_f1) * 100))
    # 如果args.col_header为真，代码会计算列头的精确度、召回率和F1分数，然后将结果打印出来。这些分数是通过对col_header_precision、col_header_recall和col_header_f1列表中的值求平均，然后乘以100得到的。
    if args.col_header:
        print("Col header: precision = %.2f; recall = %.2f; f1 = %.2f" % (
            np.mean(col_header_precision) * 100, np.mean(col_header_recall) * 100, np.mean(col_header_f1) * 100))
    # 最后，代码会计算非头单元格（即表格中的数据单元格）的精确度、召回率和F1分数，然后将结果打印出来。这些分数是通过对relation_precision、relation_recall和relation_f1列表中的值求平均，然后乘以100得到的。
    print("Non-header cell: precision = %.2f; recall = %.2f; f1 = %.2f" % (
        np.mean(relation_precision) * 100, np.mean(relation_recall) * 100, np.mean(relation_f1) * 100))
