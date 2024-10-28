#!/bin/bash
set -x


[ -z "$model_size" ] && export model_size=llama2-7B

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


if [[ "$model_size" == "llama2-7B" ]]; then
    export policy_tp=8
    export policy_pp=1
    export ppo_policy_tp=2
    export ppo_policy_pp=4
    export train_global_batch_size=128
    if [[ "$backend" == "megatron" ]]; then
        export generation_batch_size=128
        config_file=${config_dir}/llama2/rlhf_param_sync.yaml
    elif [[ "$backend" == "vllm" ]]; then
        export ENABLE_VLLM=True
        export generation_batch_size=128
        config_file=${config_dir}/llama2/vllm_param_sync.yaml
    fi
    export train_micro_batch_size=16
    export max_num_batched_tokens=65536
    export gpu_memory_utilization=0.8

    export num_gpu_policy=8
    export num_gpu_ppo_policy=8
    export free_memory_policy=True
    export free_memory_ppo_policy=True
fi

validate_param_sync=True \
policy_inference_load=${POLICY_LOAD} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_episode=${num_ppo_episode:-0} \
data_path=${DATASET_PATH} \
eval_data_path=${EVAL_DATASET_PATH} \
sample_per_episode=${sample_per_episode} \
tensorboard_dir=${TENSORBOARD_DIR} \
python tests/test_unbalanced_param_sync.py -c $config_file 2>&1 | tee ${output_dir}/log_${RANK}.log ; exit ${PIPESTATUS[0]}
