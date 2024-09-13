#!/bin/bash
set -x

[ -z "$CHATLEARN" ] && export CHATLEARN=${ROOT}/rlhf
[ -z "$ZERO_SIZE" ] && export ZERO_SIZE=8
[ -z "$LOAD" ] && export LOAD=path-to-ckpt
[ -z "$DATASET_PATH" ] && export DATASET_PATH=path-to-dataset-json
[ -z "$exp_name" ] && export exp_name=$(date +%F)-dpo
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/

output_dir=${output_dir}/${exp_name}
mkdir -p $output_dir/
log_file=$output_dir/log_${RANK}.log

cd $CHATLEARN/examples/huggingface/

source scripts/base_env.sh

generation_batch_size=4 \
num_device=$ZERO_SIZE \
zero_size=$ZERO_SIZE \
reward_dataset_path=$DATASET_PATH \
model_path=$LOAD \
python entry/train_dpo.py -c configs/qwen2/dpo.yaml 2>&1 | tee ${log_file}.log ; exit ${PIPESTATUS[0]}
