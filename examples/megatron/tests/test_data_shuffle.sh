#!/bin/bash
set -x


[ -z "$model_size" ] && export model_size=llama2-7B

export CHATLEARN=/home/data/yinzhiyu.yzy/ChatLearn
export MEGATRON=/home/data/yinzhiyu.yzy/QWen
export DATASET_PATH=/home/jiangle.jl/data/qwen2/gpt42w_ultrafeedback_shuffled.jsonl
export PYTHONPATH=${MEGATRON}/megatron_rlhf:${CHATLEARN}:${MEGATRON}:${PYTHONPATH}

source scripts/base_env.sh

backend=${1:-vllm}

[ -z "$max_new_tokens" ] && export max_new_tokens=512
[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$DATA_DIR" ] && DATA_DIR=${output_dir}/gpt/
output_dir=${output_dir}/${exp_name}
export data_checkpoint_path=${output_dir}/data_checkpoint

mkdir -p $output_dir

export max_seq_len=$(( max_new_tokens*2 ))

config_dir=${CHATLEARN}/examples/megatron/configs/

validate_param_sync=True \
policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_episode=${num_ppo_episode:-0} \
data_path=${DATASET_PATH} \
eval_data_path=${EVAL_DATASET_PATH} \
sample_per_episode=${sample_per_episode} \
python tests/test_data_shuffle.py 2>&1 ; exit ${PIPESTATUS[0]}