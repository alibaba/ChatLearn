#!/bin/bash
set -x

[ -z "$model_size" ] && export model_size=llama2-7B

# Get the directory of the current script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ${DIR}/base_env.sh

# megatron or vllm
backend=${1:-vllm}
if [[ "$backend" != "megatron" ]] && [[ "$backend" != "vllm" ]]; then
  echo "ERROR: expect megatron or vllm backend, while "$backend
  exit 1
fi


config_dir=${CHATLEARN}/examples/megatron/configs/

if [[ "$backend" == "megatron" ]]; then
  configs=${config_dir}/llama2/rlhf.yaml
else
  export ENABLE_VLLM=True
  if [ -z "$tokenizer_load" ];then
    echo "please set path to hf tokenizer for vllm backend, download from huggingface source."
    exit 1
  fi
  configs=${config_dir}/llama2/vllm_rlhf.yaml
fi

export trainer_engine=rlhf

[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$sample_per_episode" ] && sample_per_episode=1024
[ -z "$tokenizer_load" ] && export tokenizer_load=path-to-hf-tokenizer-for-vllm-backend

output_dir=${output_dir}/${exp_name}
export data_checkpoint_path=${output_dir}/data_checkpoint



if [[ "$model_size" == "llama2-7B" ]]; then
    export policy_tp=4
    export policy_pp=2

    # export policy_tp=8
    # export policy_pp=1

    export ppo_policy_tp=2
    export ppo_policy_pp=4
    # export ppo_policy_tp=4
    # export ppo_policy_pp=2

    export policy_tp=8
    export policy_pp=1

    export ppo_policy_tp=4
    export ppo_policy_pp=1
    # export ppo_policy_tp=2
    # export ppo_policy_pp=1

    export reward_tp=4
    export ppo_value_pp=1
    export train_global_batch_size=128
    if [[ "$backend" == "megatron" ]]; then
        export generation_batch_size=128
    elif [[ "$backend" == "vllm" ]]; then
        export generation_batch_size=128
    fi
    export ref_generation_batch_size=64
    export value_generation_batch_size=64
    export reward_generation_batch_size=64
    export train_micro_batch_size=16
    export max_num_batched_tokens=65536
    export gpu_memory_utilization=0.8
    # export num_gpu_ref=4
    # export num_gpu_value=4
    # export num_gpu_ppo_policy=4
    # export num_gpu_ppo_value=4
    # export free_memory_reward=True
    # export free_memory_ppo_policy=True
    # export free_memory_ppo_value=True
elif [[ "$model_size" == "llama2-13B" ]]; then
    export policy_tp=8
    export policy_pp=1
    export ppo_policy_tp=8
    export ppo_policy_pp=2
    export reward_tp=8
    export ppo_value_pp=2
    export train_global_batch_size=128
    export generation_batch_size=64
    export ref_generation_batch_size=16
elif [[ "$model_size" == "llama2-70B" ]]; then
    export policy_tp=8
    export policy_pp=1
    export ppo_policy_tp=8
    export ppo_policy_pp=4
    export reward_tp=8
    export ppo_value_pp=4
    export num_gpu_ref=16
    export num_gpu_reward=32
    export num_gpu_value=16
    export train_global_batch_size=128
    export generation_batch_size=256
    export reward_generation_batch_size=32
    export ref_generation_batch_size=32
    export value_generation_batch_size=32
    export train_micro_batch_size=4
    export max_num_batched_tokens=65536
    export gpu_memory_utilization=0.75
    export free_memory_policy=True
    export free_memory_reference=True
    export free_memory_reward=True
    export free_memory_value=True
    export free_memory_ppo_policy=True
    export free_memory_ppo_value=True
fi

mkdir -p ${output_dir}
log_file=${output_dir}/log_${RANK}.log
echo $log_file

policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_gpu=${num_gpu} \
data_path=${DATASET_PATH} \
eval_data_path=${EVAL_DATASET_PATH} \
sample_per_episode=${sample_per_episode} \
python entry/train_rlhf.py -c $configs 2>&1 | tee -a ${log_file} ; exit ${PIPESTATUS[0]}


