#!/bin/sh

dt=$(date '+%d-%m-%Y_%Hh%Mm%Ss')
log_file="res/log_$dt.log"
touch $log_file

python -u prompting/fewShot.py \
    --client_key "<your_key>" \
    --data "<your path to>/data/dev/archehr-qa.xml" \
    --keys "<your path to>/data/dev/archehr-qa_key.json" \
    --prompts_folder config/prompts \
    --model gpt-4.1-mini \
    --temperature 0.3 \
    --res_path res \
    --date "$dt" \
    --n_seeds 5 \
    --save_name "output_$dt" \
    --mode "few-shot" \
    > "$log_file" 2>&1
