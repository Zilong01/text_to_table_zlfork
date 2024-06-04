DATA_PATH=$1
ckpt=${2:-"checkpoints/checkpoint_average_best-3.pt"}

export PYTHONPATH=.


# 这段脚本使用了几种方法来评估模型的准确率。
# 总的来说，这段脚本通过计算错误格式比率和F分数来评估模型的准确率。

# 首先，它使用fairseq-interactive命令运行模型并生成预测结果。这个命令的输出被重定向到一个名为$ckpt.test_vanilla.out的文件。
printf "convert fairseq:\n"
fairseq-interactive ${DATA_PATH}/bins --path $ckpt --beam 10 --remove-bpe --buffer-size 1024 --max-tokens 4096 --user-dir src/ --task text_to_table_task  --table-max-columns 2 --unconstrained-decoding > $ckpt.test_vanilla.out < ${DATA_PATH}/test.bpe.text
# 然后，它使用convert_fairseq_output_to_text.sh脚本将fairseq-interactive的输出转换为文本格式，以便后续处理。
# 这个脚本的目的是将Fairseq模型的输出从其原始格式转换为文本格式。
bash scripts/eval/convert_fairseq_output_to_text.sh $ckpt.test_vanilla.out 

# 接着，它使用calc_data_wrong_format_ratio.py脚本计算输出数据的错误格式比率。这个脚本接收两个参数：模型的输出文件和真实的测试数据文件。它会计算模型输出中格式错误的数据的比例，这可以作为模型准确率的一个指标。
printf "Wrong format:\n"
CUDA_VISIBLE_DEVICES=2 python scripts/eval/calc_data_wrong_format_ratio.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header

# 最后，对于每个指定的度量标准（E、c、BS-scaled），它使用calc_data_f_score.py脚本计算F分数。F分数是精确率和召回率的调和平均，是评估模型准确率的常用指标。这个脚本接收四个参数：模型的输出文件、真实的测试数据文件、是否有行头以及度量标准。
for metric in E c BS-scaled; do
  printf "$metric metric:\n"
  python scripts/eval/calc_data_f_score.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header --metric $metric
done