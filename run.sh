#!/bin/sh

dt=$(date '+%d-%m-%Y_%Hh%Mm%Ss')
log_file="res/log_$dt.log"
touch $log_file

python -u src/few_shot.py \
    --client_key "YOUR_KEY" \
    --data /Users/armandviolle/Developer/challenge-BioNLP/data/dev/archehr-qa.xml \
    --keys /Users/armandviolle/Developer/challenge-BioNLP/data/dev/archehr-qa_key.json \
    --prompts_folder config \
    --model gpt-4.1-mini \
    --res_path res \
    --date "$dt" \
    --n_seeds 5 \
    --save_name "output_$dt" \
    --mode "few-shot" \
    > "$log_file" 2>&1
