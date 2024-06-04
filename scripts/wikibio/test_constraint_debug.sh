DATA_PATH=$1
ckpt=${2:-"checkpoints/checkpoint_average_best-3.pt"}

export PYTHONPATH=.

for metric in E c BS-scaled; do # 每个评估标准
  printf "$metric metric:\n"
  CUDA_VISIBLE_DEVICES=3 python -m debugpy --listen 52435 --wait-for-client scripts/eval/calc_data_f_score.py $ckpt.test_constrained.out.text ${DATA_PATH}/test.data --row-header --metric $metric # 只有行表头
done