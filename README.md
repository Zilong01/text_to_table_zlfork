**Update 01/12/2023**: for BERTScore evaluation, using `transformers>4.17.0` will lead to different results as in the paper. If you have difficulty replicating the BERTScore results, please try downgrading to `transformers<4.17.0`. In my experiments I used `transformers==3.1.0`. For details please refer to https://github.com/shirley-wu/text_to_table/issues/7

**Update 12/23/2023**: 更新代码在全部数据集，全部操作下的执行脚本命令。
---

## Introduction

This repository is for our ACL2022 paper: [Text-to-Table: A New Way of Information Extraction](https://arxiv.org/abs/2109.02707).

## Requirements

Training requires `fairseq==v0.10.2`, and evaluation requires `sacrebleu==v2.0.0 bert_score==v0.3.11`

Or you can directly install by `pip install -r requirements.txt`.

Note: to avoid potential incompatibility, your fairseq version should be **exactly v0.10.2**, and your python version should be **<3.9**

## Dataset

You can download the four datasets from [Google Drive](https://drive.google.com/file/d/1zTfDFCl1nf_giX7IniY5WbXi9tAuEHDn/view?usp=sharing). If you are interested in preprocessing original table-to-text datasets into our text-to-table datasets, please check [data_preprocessing](data_preprocessing/).

For preprocessing, we use `fairseq` for BPE and binarization. You need to first download a BART model [here](https://github.com/pytorch/fairseq/tree/main/examples/bart), and then use `scripts/preprocess.sh` to preprocess the data. The script has two arguments: the first is the data path and the second is the bart model path, e.g.,
```bash
bash scripts/preprocess.sh data/rotowire/ bart.base/
bash scripts/preprocess.sh data/wikitabletext/ bart.base/
bash scripts/preprocess.sh data/wikibio/ bart.base/
```
then you'll have BPE-ed files under `data/rotowire` and binary files under `data/rotowire/bins`.

## Training

For each dataset, use `scripts/dataset-name/train_vanilla.sh` to train a vanilla seq2seq model, and use `scripts/dataset-name/train_vanilla.sh` to train a HAD model. The training scripts have three arguments: the first is the data path (NOTE: it's not the path to the binary files), the second is the bart model path, and the third is the saving directory (optional argument, default is `checkpoints/`). Example use:
```bash


```

下面两个数据集在单个gpu上运行
wzl-e2e
```bash
bash scripts/e2e/train_vanilla.sh data/e2e/ bart.base/ checkpoints/e2e/
bash scripts/e2e/train_had.sh data/e2e/ bart.base/ checkpoints/e2e/
```

wzl-wikitabletext
```bash
# WikiTableText，数据集非常小，因此我们使用 5 个种子（1、10、20、30、40）
seeds1=(1 10 20 30 40)
bash scripts/wikitabletext/train_vanilla.sh data/wikitabletext/ bart.base/ checkpoints/wikitabletext/vanilla/
bash scripts/wikitabletext/train_had.sh data/wikitabletext/ bart.base/ checkpoints/wikitabletext/had/
```

~~在多个GPU上运行（这里设置为2块，原文为8块）~~
多GPU报错，改为单GPU运行

wzl-rotowire
```bash
bash scripts/rotowire/train_vanilla.sh data/rotowire/ bart.base/ checkpoints/rotowire/vanilla/
bash scripts/rotowire/train_had.sh data/rotowire/ bart.base/ checkpoints/rotowire/had/
```


wzl-wikibio
```bash
bash scripts/wikibio/train_vanilla.sh data/wikibio/ bart.base/ checkpoints/wikibio/vanilla/
bash scripts/wikibio/train_had.sh data/wikibio/ bart.base/ checkpoints/wikibio/had/
```


or

```bash
bash scripts/rotowire/train_had.sh data/rotowire/ bart.base/ checkpoints/rotowire/
```

**Note**: for simplicity, the saving directory for all scripts are set to `checkpoints/` by default. However, if you want to run multiple training experiments, remember to change the saving directory argument. Otherwise the experiments may override the saved checkpoints and lead to unexpected behavior.

Additionally, for Rotowire and WikiTableText, the datasets are very small, so we run experiments with 5 seeds (1, 10, 20, 30, 40) and report the average numbers. Scripts under `scripts/rotowire` and `scripts/wikitabletext` have the seed as the fourth argument.

Rotowire and WikiBio experiments are run on 8 GPUs. E2E and WikiTableText experiments are run on 1 GPU.

You'll need GPUs that supports `--fp16` (such as V100). If not, please remove the `--fp16` option in the scripts.

## Inference and Evaluation

For each dataset, use `scripts/dataset-name/test_vanilla.sh` to test with vanilla decoding, and use `scripts/dataset-name/test_constraint.sh` to test with table constraint. The test scripts have two arguments: the first is the data path and the second is the checkpoint path (by default it is where your saved checkpoint goes to), e.g.,
```bash
bash scripts/rotowire/test_constraint.sh data/rotowire/ 
```

wzl-e2e
```bash
bash scripts/e2e/test_constraint.sh data/e2e/ checkpoints/e2e/had_yiwen/checkpoint_average_best-3.pt
bash scripts/e2e/test_vanilla.sh data/e2e/ checkpoints/e2e/vanilla/checkpoint_average_best-3.pt
```

wzl-wikitabletext
```bash
bash scripts/wikitabletext/test_constraint.sh data/wikitabletext/ checkpoints/wikitabletext/had/checkpoint_average_best-3.pt
bash scripts/wikitabletext/test_vanilla.sh data/wikitabletext/ checkpoints/wikitabletext/vanilla/checkpoint_average_best-3.pt

```

wzl-rotowire
```bash
bash scripts/rotowire/test_constraint.sh data/rotowire/ checkpoints/rotowire/had/checkpoint_average_best-3.pt
bash scripts/rotowire/test_vanilla.sh data/rotowire/ checkpoints/rotowire/vanilla/checkpoint_average_best-3.pt
```


wzl-wikibio
```bash
bash scripts/wikibio/test_constraint.sh data/wikibio/ checkpoints/wikibio/had/checkpoint_average_best-3.pt
bash scripts/wikibio/test_vanilla.sh data/wikibio/ checkpoints/wikibio/vanilla/checkpoint_average_best-3.pt
```

Similar to training, you'll need GPUs that supports `--fp16`. If not, please remove `--fp16` in the script.
