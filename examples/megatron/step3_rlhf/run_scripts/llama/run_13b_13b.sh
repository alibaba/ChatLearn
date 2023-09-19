#!/bin/bash

set -x

export model_size=13B
export policy_tp=8
export ppo_policy_pp=2
export reward_tp=8
export ppo_reward_pp=2
source run_scripts/llama/base_env.sh


cd ${CHATLEARN}/examples/megatron/step3_rlhf
if [ -z "${exp_name}" ]; then
    export exp_name=$(date +%F)_llama-rlhf-${model_size}-${model_size}
fi

[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR=${CHATLEARN}/output/step3_rlhf/
[ -z "$LOG_DIR" ] && LOG_DIR=${OUTPUT_DIR}/logs/${exp_name}
[ -z "$TENSORBOARD_DIR" ] && TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard/${exp_name}
[ -z "$SAVE_DIR" ] && SAVE_DIR=${OUTPUT_DIR}/save_model/${exp_name}

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
python train_rlhf.py -c configs/llama/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt


set +x
