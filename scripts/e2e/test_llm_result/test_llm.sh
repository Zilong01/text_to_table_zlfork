# bash scripts/e2e/test_llm_result/test_llm.sh data/e2e/ 100
DATA_PATH=$1
MY_LLM_RESULT="/workspace/wzl/datamining/text_to_table/scripts/e2e/test_llm_result/e2e_test_table.txt" # llm输出的结果
# ckpt=${2:-"checkpoints/checkpoint_average_best-3.pt"}

export PYTHONPATH=.

n_rows=$2  # 添加n_rows参数

# 使用sed命令来截取test.data文件的前n_rows行
# 将截取后的内容保存到临时文件test.data.truncated中
head -n "$n_rows" ${DATA_PATH}/test.data > ${DATA_PATH}/test.data.truncated

# 使用截取后的文件test.data.truncated来计算错误格式比率
printf "Wrong format:\n"
CUDA_VISIBLE_DEVICES=2 python scripts/eval/calc_data_wrong_format_ratio.py $MY_LLM_RESULT ${DATA_PATH}/test.data.truncated --row-header

# 调试使用：
# python3 -m debugpy --listen 52435 --wait-for-client scripts/eval/calc_data_wrong_format_ratio.py /workspace/wzl/datamining/text_to_table/scripts/e2e/test_llm_result/e2e_test_table.txt  /workspace/wzl/datamining/text_to_table/data/e2e/test.data.truncated --row-header


# 对于每个度量标准，使用截取后的文件test.data.truncated来计算F分数
for metric in E c BS-scaled; do
  printf "$metric metric:\n"
  python scripts/eval/calc_data_f_score.py $MY_LLM_RESULT ${DATA_PATH}/test.data.truncated --row-header --metric $metric
done

# debug模式
# python3 -m debugpy --listen 52435 --wait-for-client scripts/eval/calc_data_f_score.py /workspace/wzl/datamining/text_to_table/scripts/e2e/test_llm_result/e2e_test_table.txt  /workspace/wzl/datamining/text_to_table/data/e2e/test.data.truncated --row-header --metric E




# 这段脚本使用了几种方法来评估模型的准确率。
# 总的来说，这段脚本通过计算错误格式比率和F分数来评估模型的准确率。

# 这个脚本的目的是将Fairseq模型的输出从其原始格式转换为文本格式。

# 接着，它使用calc_data_wrong_format_ratio.py脚本计算输出数据的错误格式比率。这个脚本接收两个参数：模型的输出文件和真实的测试数据文件。它会计算模型输出中格式错误的数据的比例，这可以作为模型准确率的一个指标。
# printf "Wrong format:\n"
# CUDA_VISIBLE_DEVICES=2 python scripts/eval/calc_data_wrong_format_ratio.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header

# # 最后，对于每个指定的度量标准（E、c、BS-scaled），它使用calc_data_f_score.py脚本计算F分数。F分数是精确率和召回率的调和平均，是评估模型准确率的常用指标。这个脚本接收四个参数：模型的输出文件、真实的测试数据文件、是否有行头以及度量标准。
# for metric in E c BS-scaled; do
#   printf "$metric metric:\n"
#   python scripts/eval/calc_data_f_score.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header --metric $metric
# done