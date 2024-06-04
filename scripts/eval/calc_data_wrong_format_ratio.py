import argparse

from table_utils import (
    extract_table_by_name,
    parse_text_to_table,
    is_empty_table,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp') # 这是一个位置参数，表示预测数据的文件路径。位置参数是必须要提供的参数，如果在命令行中没有提供，程序会报错。
    parser.add_argument('tgt') # 这也是一个位置参数，表示目标数据的文件路径。
    parser.add_argument('--row-header', default=False, action="store_true") # 这是一个可选参数，表示是否有行头。如果在命令行中提供了这个参数，那么它的值为True，否则为False。
    parser.add_argument('--col-header', default=False, action="store_true") # 这是一个可选参数，表示是否有列头。如果在命令行中提供了这个参数，那么它的值为True，否则为False。
    parser.add_argument('--table-name', default=None) # 这是一个可选参数，表示表格的名称。如果在命令行中没有提供这个参数，那么它的值为None。
    args = parser.parse_args()
    assert args.row_header or args.col_header
    print("Args", args)
    return args


if __name__ == "__main__":
    args = parse_args()

    hyp_data = []
    with open(args.hyp) as f: # 打开预测数据文件
        for line in f:
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            d = parse_text_to_table(line, strict=True)
            hyp_data.append(d)
            if d is None:
                print("预测结果中出现了空,Wrong format: ", line)
    tgt_data = []
    with open(args.tgt) as f: # 打开目标数据文件
        for line in f:
            line = line.strip()
            if args.table_name is not None:
                line = extract_table_by_name(line, args.table_name)
            tgt_data.append(parse_text_to_table(line, strict=True))

    empty_tgt = 0
    wrong_format = 0
    for hyp_table, tgt_table in zip(hyp_data, tgt_data): # hyp_data和tgt_data是每行文本转换成的表格列表
        if is_empty_table(tgt_table, args.row_header, args.col_header):
            empty_tgt += 1
        elif hyp_table is None:
            # 如果预测表格是None，那么认为预测表格的格式错误，wrong_format计数器加1。
            wrong_format += 1
            print("Wrong format: ", hyp_table, tgt_table)

    valid_tgt = len(hyp_data) - empty_tgt
    print("Wrong format: %d / %d (%.2f%%)" % (wrong_format, valid_tgt, wrong_format / valid_tgt * 100))
