#!/bin/bash
set -x

# set path
export CHATLEARN=$(pwd)
export model_path="${CHATLEARN}/Qwen3-8B"
export exp_name=$(date +%F)-qwen3-grpo
export output_dir=${CHATLEARN}/output/${exp_name}
export train_data_path=${CHATLEARN}/dataset/MATH-lighteval/train.json
export eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test.json
export data_checkpoint_path=${output_dir}/data_checkpoint_path/
log_file=$output_dir/log_${RANK}.log
mkdir -p $output_dir/
export log_dir=${output_dir}
export wandb_dir=${output_dir}

cd $CHATLEARN/examples/fsdp/
source scripts/base_env.sh

# Env setup
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_DEDUP_LOGS=1
export NCCL_NVLS_ENABLE=0

# log setup
export enable_wandb=False
export wandb_project="grpo-exp"
export WANDB_API_KEY="wandb-api-key"

#VLLM setup
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1

export tensor_model_parallel_size=2
export policy_temperature=1.0
export policy_top_p=1.0
export policy_top_k=-1
export policy_eval_temperature=0.6
export policy_eval_top_p=0.95
export policy_eval_top_k=20

export seq_length=2048
export max_new_tokens=2048
export max_seq_len_to_capture=2348
export num_inference_per_prompt=32
export train_global_batch_size=2048 
export sample_per_episode=2048
export vllm_generation_batch_size=128
export train_micro_batch_size=8
export gpu_memory_utilization=0.85

export enable_eval_before_training=False
export eval_episode_interval=5
export save_episode_interval=5
# for qwen3 where enable_thinking
export enable_thinking=False

python entry/train_grpo.py -c configs/grpo/grpo.yaml 2>&1 | tee ${log_file}.log ; exit ${PIPESTATUS[0]}
