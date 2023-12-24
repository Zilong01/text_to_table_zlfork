# 测试本地加载模型
from bert_score import BERTScorer

# 设置模型路径为本地路径
model_type_local = "/workspace/wzl/datamining/text_to_table/wzl/roberta-large/"

# bert_scorer = bert_score.BERTScorer(lang="en", rescale_with_baseline=True) # 原代码：这里无法下载模型，导致出错
bert_scorer = BERTScorer(model_type=model_type_local, num_layers=8, lang="en", rescale_with_baseline=True)
