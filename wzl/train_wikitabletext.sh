# 以默认种子值运行
bash scripts/wikitabletext/train_vanilla.sh data/wikitabletext/ bart.base/ checkpoints/wikitabletext/vanilla/
bash scripts/wikitabletext/train_had.sh data/wikitabletext/ bart.base/ checkpoints/wikitabletext/had/

bash scripts/rotowire/train_vanilla.sh data/rotowire/ bart.base/ checkpoints/rotowire/vanilla/
bash scripts/rotowire/train_had.sh data/rotowire/ bart.base/ checkpoints/rotowire/had/


bash scripts/wikibio/train_vanilla.sh data/wikibio/ bart.base/ checkpoints/wikibio/vanilla/
bash scripts/wikibio/train_had.sh data/wikibio/ bart.base/ checkpoints/wikibio/had/