import numpy as np

SEP = "|"


# parse_table_to_text函数将一个二维列表（表格）转换为文本。它首先将表格中的所有元素转换为字符串，然后检查是否有元素包含分隔符（默认为"|"）。如果有，并且没有提供转义字符，那么会抛出断言错误。如果提供了转义字符，那么会将包含分隔符的元素中的分隔符替换为转义字符。然后，它将表格转换为文本，每行的元素用分隔符连接，每行之间用换行符连接。如果one_line参数为True，那么会将换行符替换为""。
def parse_table_to_text(table, one_line=False, escape_token=None):
    table = [[str(xx) for xx in x] for x in table]
    has_sep_inside = any([SEP in xx for x in table for xx in x])
    if escape_token is None:
        assert not has_sep_inside
    elif has_sep_inside:
        # print("Escape! '{}' -> '{}'. Data: {}".format(SEP, escape_token, table))
        table = [[xx.replace(SEP, escape_token) for xx in x] for x in table]
    text = "\n".join([("{} ".format(SEP) + " {} ".format(SEP).join(x) +
                       " {}".format(SEP)) for x in table])
    if one_line:
        text = text.replace("\n", " <NEWLINE> ")
    return text

# parse_text_to_table函数将文本转换为二维列表（表格）。它首先将文本中的"<NEWLINE>"替换为换行符，然后按行分割文本，每行的元素用分隔符分割。如果strict参数为False，那么会将所有行的元素数量统一为第一行的元素数量，不足的部分用空字符串填充。最后，尝试将列表转换为numpy数组，如果转换失败，并且strict参数为True，那么会抛出断言错误。
def parse_text_to_table(text, strict=False):
    text = text.replace(" <NEWLINE> ", "\n").strip()
    text = text.replace("<NEWLINE>", "\n").strip() # 防止有些文本中的"<NEWLINE>"没有空格，或者有些文本中的"<NEWLINE>"在末尾
    data = []
    for line in text.splitlines(): # 上面换成换行符，这里就可以逐行比较了
        line = line.strip()
        if not line.startswith(SEP): # "|"
            line = SEP + line
        if not line.endswith(SEP):
            line = line + SEP
        data.append([x.strip() for x in line[1:-1].split(SEP)])
    if not strict and len(data) > 0: # 确保data中的所有子列表（即每一行）的长度都是一样的。
        n_col = len(data[0])
        data = [d[:n_col] for d in data]
        data = [d + ["", ] * (n_col - len(d)) for d in data] # 这个操作会在每一行的末尾添加足够数量的空字符串，使得每一行的长度都等于n_col。这个操作的结果是一个新的列表，它的每一行的长度都等于n_col。
    try:
        data = np.array(data, dtype=np.str)
    except:
        assert strict
        data = None
    return data # 返回一个numpy数组，每一个元素是一个array，包含列名和行名，都是字符串。

# extract_table_by_name函数从文本中提取指定名称的表格。它首先按行分割文本，然后查找包含指定名称的行，从该行之后的行开始，直到遇到以":"结尾的行为止，将这些行组成一个新的表格。
def extract_table_by_name(text, name):
    lines = [line.strip() for line in text.replace(" <NEWLINE> ", "\n").strip().splitlines()]
    if name + ":" not in lines:
        return ""
    table = []
    for line in lines[lines.index(name + ":") + 1:]:
        if line.endswith(":"):
            break
        table.append(line.strip())
    return " <NEWLINE> ".join(table)

# is_empty_table函数检查一个表格是否为空。如果表格为None，或者表格的形状不是2，或者表格的行数或列数小于2（取决于row_name和col_name参数），那么返回True，否则返回False。
def is_empty_table(table, row_name: bool, col_name: bool):
    if table is None:
        return True
    if len(table.shape) != 2:
        assert table.size == 0
        return True
    row, col = table.shape
    if row_name and col < 2:
        return True
    if col_name and row < 2:
        return True
    return False
