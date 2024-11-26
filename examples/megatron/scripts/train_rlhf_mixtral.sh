#!/bin/bash
set -x

[ -z "$model_size" ] && export model_size=mixtral-8x7B

# Get the directory of the current script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ${DIR}/base_env.sh

# megatron
# TODO: support vllm
backend=${1:-megatron}
if [[ "$backend" != "megatron" ]]; then
  echo "ERROR: expect megatron backend for Mixtral models, while current backend is "$backend
  exit 1
fi


config_dir=${CHATLEARN}/examples/megatron/configs/

if [[ "$backend" == "megatron" ]]; then
  configs=${config_dir}/mixtral/rlhf.yaml
else
  export ENABLE_VLLM=True
  if [ -z "$tokenizer_load" ];then
    echo "please set path to hf tokenizer for vllm backend, download from huggingface source."
    exit 1
  fi
  configs=${config_dir}/mixtral/vllm_rlhf.yaml
fi

export trainer_engine=rlhf

[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$sample_per_episode" ] && sample_per_episode=1024
[ -z "$tokenizer_load" ] && export tokenizer_load=path-to-hf-tokenizer-for-vllm-backend

output_dir=${output_dir}/${exp_name}
export data_checkpoint_path=${output_dir}/data_checkpoint


if [[ "$model_size" == "mixtral-8x7B" ]]; then
    export policy_tp=1
    export policy_ep=8
    export ppo_policy_pp=4
    export reward_tp=1
    export reward_ep=8
    export ppo_value_pp=4
    export ref_pp=4
    export policy_pp=4
    export reward_pp=4
    export value_pp=4
    export train_global_batch_size=32
    export generation_batch_size=8
    export ref_generation_batch_size=8
    export train_micro_batch_size=1
    export policy_recompute_activations=True
    export policy_moe_layer_recompute=True
    export value_recompute_activations=True
    export value_moe_layer_recompute=True
    export free_memory_policy=True
    export free_memory_reference=True
    export free_memory_reward=True
    export free_memory_value=True
    export free_memory_ppo_policy=True
    export free_memory_ppo_value=True
    export seq_length=2048
    export max_new_tokens=1024
else
  echo "Unrecognized model_size ${model_size}, choose from 'mixtral-8x7B'."
  exit -1
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
