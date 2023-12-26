DATA_PATH=$1
ckpt=${2:-"checkpoints/checkpoint_average_best-3.pt"}

export PYTHONPATH=.

CUDA_VISIBLE_DEVICES=2 fairseq-interactive ${DATA_PATH}/bins --path $ckpt --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 8192 --max-len-b 1024 --user-dir src/ --task text_to_table_task  --table-max-columns 2 --unconstrained-decoding > $ckpt.test_vanilla.out < ${DATA_PATH}/test.bpe.text
CUDA_VISIBLE_DEVICES=2 bash scripts/eval/convert_fairseq_output_to_text.sh $ckpt.test_vanilla.out

printf "Wrong format:\n"
CUDA_VISIBLE_DEVICES=2 python scripts/eval/calc_data_wrong_format_ratio.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header
for metric in E c BS-scaled; do
  printf "$metric metric:\n"
  CUDA_VISIBLE_DEVICES=2 python scripts/eval/calc_data_f_score.py $ckpt.test_vanilla.out.text ${DATA_PATH}/test.data --row-header --metric $metric
done