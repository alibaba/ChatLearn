#!/bin/bash

set -x

export model_size=70B
export policy_tp=8
export ppo_policy_pp=4
export reward_tp=8
export ppo_value_pp=4


source run_scripts/llama2/base_env.sh


cd ${CHATLEARN}/examples/megatron/step3_rlhf
if [ -z "${exp_name}" ]; then
    export exp_name=$(date +%F)_llama-rlhf-${model_size}-${model_size}
fi

[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR=${CHATLEARN}/output/step3_rlhf/
[ -z "$LOG_DIR" ] && LOG_DIR=${OUTPUT_DIR}/logs/${exp_name}
[ -z "$TENSORBOARD_DIR" ] && TENSORBOARD_DIR=${OUTPUT_DIR}/tensorboard/${exp_name}
[ -z "$SAVE_DIR" ] && SAVE_DIR=${OUTPUT_DIR}/save_model/${exp_name}
[ -z "$sample_per_episode" ] && sample_per_episode=1024

mkdir -p ${LOG_DIR}

export policy_recompute_activations=True
export policy_recompute_granularity=full
export value_recompute_activations=True
export value_recompute_granularity=full
export batch_generation_min_prompt_length=32
export num_device_ref=16
export num_device_reward=32
export num_device_value=16
export ref_pp=2
export reward_pp=2
export inference_batch_times_seqlen_threshold=16384

enable_lora_policy=True \
enable_lora_value=True \
policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_device=${num_device} \
log_dir=${LOG_DIR} \
tensorboard_dir=${TENSORBOARD_DIR} \
save_dir=${SAVE_DIR} \
data_path=${DATASET_PATH} \
sample_per_episode=${sample_per_episode} \
train_global_batch_size=128 \
generation_batch_size=64 \
ref_generation_batch_size=16 \
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt


set +x
