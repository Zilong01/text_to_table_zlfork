DATA_PATH=$1
ckpt=${2:-"checkpoints/checkpoint_average_best-3.pt"}

export PYTHONPATH=.

fairseq-interactive ${DATA_PATH}/bins --path $ckpt --beam 5 --remove-bpe --buffer-size 1024 --max-tokens 4096 --user-dir src/ --task text_to_table_task  --table-max-columns 2 > $ckpt.test_constrained.out < ${DATA_PATH}/test.bpe.text
bash scripts/eval/convert_fairseq_output_to_text.sh $ckpt.test_constrained.out

printf "Wrong format:\n"
CUDA_VISIBLE_DEVICES=2 python scripts/eval/calc_data_wrong_format_ratio.py $ckpt.test_constrained.out.text ${DATA_PATH}/test.data --row-header
for metric in E c BS-scaled; do
  printf "$metric metric:\n"
  python scripts/eval/calc_data_f_score.py $ckpt.test_constrained.out.text ${DATA_PATH}/test.data --row-header --metric $metric
done