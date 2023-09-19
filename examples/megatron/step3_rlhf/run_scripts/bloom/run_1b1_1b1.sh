#!/bin/bash

set -x

export model_size=1B1
export policy_tp=4
export ppo_policy_pp=1
export reward_tp=4
export ppo_reward_pp=1

source run_scripts/bloom/base_env.sh

cd ${CHATLEARN}/examples/megatron/step3_rlhf

if [ -z "${exp_name}" ]; then
    export exp_name=$(date +%F)_bloom-rlhf-${model_size}-${model_size}
fi

[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR=${CHATLEARN}/output/step3_rlhf/
LOG_DIR=${OUTPUT_DIR}/logs/${exp_name}
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard/${exp_name}
SAVE_DIR=${OUTPUT_DIR}/save_model/${exp_name}

mkdir -p ${LOG_DIR}

policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
vocab_file=${VOCAB_FILE} \
num_device=${num_device} \
log_dir=${LOG_DIR} \
tensorboard_dir=${TENSORBOARD_DIR} \
save_dir=${SAVE_DIR} \
data_path=${DATASET_PATH} \
sample_per_episode=1024 \
train_global_batch_size=128 \
generation_batch_size=64 \
ref_generation_batch_size=16 \
python train_rlhf.py -c configs/bloom/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt

set +x
