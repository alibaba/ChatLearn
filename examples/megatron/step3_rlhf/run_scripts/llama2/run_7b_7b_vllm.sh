#!/bin/bash
set -x

export model_size=7B
export policy_tp=8
export ppo_policy_pp=1
export reward_tp=8
export ppo_value_pp=1

source run_scripts/llama2/base_env.sh

cd ${CHATLEARN}/examples/megatron/step3_rlhf

if [ -z "${exp_name}" ]; then
   export exp_name=$(date +%F)_llama-rlhf-${model_size}-${model_size}
fi

[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR=${CHATLEARN}/output/step3_rlhf/
LOG_DIR=${OUTPUT_DIR}/logs/${exp_name}
TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard/${exp_name}
SAVE_DIR=${OUTPUT_DIR}/save_model/${exp_name}

mkdir -p ${LOG_DIR}

policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_device=${num_device} \
log_dir=${LOG_DIR} \
tensorboard_dir=${TENSORBOARD_DIR} \
save_dir=${SAVE_DIR} \
data_path=${DATASET_PATH} \
sample_per_episode=1024 \
train_global_batch_size=128 \
generation_batch_size=64 \
ref_generation_batch_size=16 \
python train_vllm_rlhf.py -c configs/llama2/vllm_rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_vllm_${RANK}.txt ; exit ${PIPESTATUS[0]}

