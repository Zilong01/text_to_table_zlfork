import sys

import numpy as np


def get_hypothesis(inp, oup):
    """
    读取输入文件，提取以“H-”开头的假设行，
    根据它们的ID对它们进行排序，并将排序后的文本写入输出文件。

    Args:
        inp (str): Path to the input file.
        oup (str): Path to the output file.
    """
    _, inp, oup = sys.argv

    with open(inp) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines if x.startswith("H-")]

    ids = [int(x.split("H-")[1].split()[0]) for x in lines]
    assert sorted(ids) == list(range(len(ids)))
    texts = [x.split("\t")[-1] for x in lines]

    with open(oup, 'w') as f:
        for i in np.lexsort((np.arange(len(ids)), ids)):
            # preserve order for multi beam
            # can also use `kind='stable'`; but don't wanna do that
            f.write(texts[i] + '\n')
