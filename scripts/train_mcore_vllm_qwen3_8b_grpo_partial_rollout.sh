#!/bin/bash
set -x

# Tested on 8xH20 with 96G VRAM
export RAY_CGRAPH_get_timeout=200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=1
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1

export CHATLEARN=$(pwd)
export MEGATRON_PATH=${CHATLEARN}/../Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624
export PYTHONPATH=${CHATLEARN}:${MEGATRON_PATH}:${PYTHONPATH}
source scripts/base_env.sh

hf_ckpt_path=${CHATLEARN}/pretrained_models/Qwen3-8B
mcore_ckpt_path=${CHATLEARN}/pretrained_models/Qwen3-8B-to-mcore

exp_name="test_qwen3_8b"
export output_dir=${CHATLEARN}/output/${exp_name}
mkdir -p $output_dir/
export log_dir=${output_dir}/logs
mkdir -p $log_dir
log_file=$log_dir/${exp_name}_rank${RANK}.log

python chatlearn/entrypoint.py grpo --config-file template/grpo_megatron.yaml \
        runtime_args.exp_name=${exp_name} \
        runtime_args.partial_rollout=True \
        runtime_args.log_args_dict.enable_tensorboard=True \
        runtime_args.train_backend=megatron \
        runtime_args.data_path=${CHATLEARN}/dataset/MATH-lighteval/train.json \
        runtime_args.eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test.json \
        runtime_args.output_dir=${CHATLEARN}/output/${exp_name} \
        runtime_args.num_episode=50 \
        runtime_args.sample_per_episode=2048 \
        runtime_args.train_global_batch_size=2048 \
        runtime_args.train_micro_batch_size=1 \
        runtime_args.save_episode_interval=1000000 \
        runtime_args.log_args_dict.enable_tensorboard=true \
        runtime_args.log_args_dict.tensorboard_dir=${output_dir}/tensorboard \
        runtime_args.eval_episode_interval=1 \
        runtime_args.enable_eval_before_training=true \
        models.policy_trainer.num_gpu=${num_device} \
        models.policy_trainer.packing=true \
        models.policy_trainer.max_token_in_packing=8192 \
        models.policy_trainer.bf16=true \
        models.policy_trainer.sequence_parallel=true \
        models.policy_trainer.use_distributed_optimizer=true \
        models.policy_trainer.recompute_granularity=null \
        models.policy_trainer.tensor_model_parallel_size=2 \
        models.policy_trainer.pipeline_model_parallel_size=1 \
        models.policy_trainer.generation_batch_size=512 \
        models.policy_trainer.load=${mcore_ckpt_path} \
        models.policy_trainer.optimizer.lr=2e-6 \
        models.policy_trainer.optimizer.min_lr=2e-6 \
        models.policy_trainer.pos_clip_ratio=0.2 \
        models.policy_trainer.neg_clip_ratio=0.2 \
        models.reward.generation_batch_size=128 \
        models.policy.load=${hf_ckpt_path} \
        models.policy.generation_batch_size=256 \
        models.policy.tensor_model_parallel_size=1 \
        models.policy.max_prompt_tokens_length=1024 \
        models.policy.max_response_tokens_length=2048 \
        models.policy.num_inference_per_prompt=32 \
        models.policy.gpu_memory_utilization=0.75 \
        models.policy.enable_thinking=False \
        models.rollout_manager.max_rollout_round=2 \
        models.rollout_manager.rollout_ratio=[0.5,0.5] \
        models.rollout_manager.mini_response_per_prompt=8 \
        2>&1 | tee ${log_file} ; exit ${PIPESTATUS[0]}