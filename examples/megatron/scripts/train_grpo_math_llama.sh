#!/bin/bash
set -x

[ -z "$model_size" ] && export model_size=llama2-7B

# Get the directory of the current script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ${DIR}/base_env.sh

export trainer_engine=online_dpo
model_name=llama

export train_to_compare_num_responses=8
export num_inference_per_prompt=8
export ENABLE_VLLM=True
if [ -z "$tokenizer_load" ];then
  echo "please set path to hf tokenizer for vllm backend, download from huggingface source."
  exit 1
fi
configs=$CHATLEARN/examples/megatron/configs/llama2/grpo_math_vllm.yaml

[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$sample_per_episode" ] && sample_per_episode=1024
[ -z "$tokenizer_load" ] && export tokenizer_load=path-to-hf-tokenizer-for-vllm-backend
output_dir=${output_dir}/${exp_name}
export data_checkpoint_path=${output_dir}/data_checkpoint
mkdir -p $output_dir
log_file=${output_dir}/log_${RANK}.log

export prompt_key=prompt

export policy_tp=8
export ppo_policy_pp=1
export reward_tp=8
export ppo_value_pp=1

trainer_engine=grpo \
policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_gpu=${num_gpu} \
data_path=${DATASET_PATH} \
eval_data_path=${EVAL_DATASET_PATH} \
sample_per_episode=${sample_per_episode} \
train_global_batch_size=128 \
generation_batch_size=128 \
ref_generation_batch_size=16 \
train_micro_batch_size=8 \
python entry/train_grpo_math.py -c $configs 2>&1 | tee -a ${log_file} ; exit ${PIPESTATUS[0]}

