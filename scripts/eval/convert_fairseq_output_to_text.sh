x=$1
printf "run get_hypothesis.py: \n"
python $( dirname $0 )/get_hypothesis.py $x $x.hyp
printf "run gpt2_decode.py: \n"
python $( dirname $0 )/gpt2_decode.py $x.hyp $x.text
