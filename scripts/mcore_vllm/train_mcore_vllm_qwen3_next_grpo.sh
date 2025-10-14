#!/bin/bash
set -x

export RAY_CGRAPH_get_timeout=200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_num_server_call_thread=1
export RAY_DEDUP_LOGS=0
export VLLM_USE_RAY_SPMD_WORKER=1
export VLLM_USE_RAY_COMPILED_DAG=1

export CHATLEARN=$(pwd)
export MEGATRON_PATH=${CHATLEARN}/../Pai-Megatron-Patch/backends/megatron/Megatron-LM-250908
export PYTHONPATH=${CHATLEARN}:${MEGATRON_PATH}:${PYTHONPATH}
source scripts/base_env.sh

hf_ckpt_path=${CHATLEARN}/pretrained_models/Qwen3-Next-80B-A3B-Instruct 
mcore_ckpt_path=${CHATLEARN}/pretrained_models/Qwen3-Next-80B-A3B-Instruct-to-mcore


exp_name="test_qwen3_next_grpo"
export output_dir=${CHATLEARN}/output/${exp_name}
mkdir -p $output_dir/
export log_dir=${output_dir}/logs
mkdir -p $log_dir
log_file=$log_dir/${exp_name}_rank${RANK}.log

python chatlearn/entrypoint.py grpo --config-file template/grpo_megatron.yaml \
        runtime_args.exp_name=${exp_name} \
        runtime_args.log_args_dict.enable_tensorboard=True \
        runtime_args.train_backend=megatron \
        runtime_args.rollout_backend=sglang \
        runtime_args.data_path=${CHATLEARN}/dataset/MATH-lighteval/train.json \
        runtime_args.eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test.json \
        runtime_args.output_dir=${CHATLEARN}/output/${exp_name} \
        runtime_args.num_episode=200 \
        runtime_args.sample_per_episode=256 \
        runtime_args.train_global_batch_size=256 \
        runtime_args.train_micro_batch_size=1 \
        runtime_args.save_episode_interval=1000000 \
        runtime_args.log_args_dict.enable_tensorboard=True \
        runtime_args.log_args_dict.tensorboard_dir=${output_dir}/tensorboard \
        runtime_args.log_args_dict.enable_wandb=False \
        runtime_args.eval_episode_interval=5 \
        runtime_args.enable_eval_before_training=False \
        models.policy_trainer.num_gpu=${num_device} \
        models.policy_trainer.packing=False \
        models.policy_trainer.trust_remote_code=True \
        models.policy_trainer.max_token_in_packing=3072 \
        models.policy_trainer.bf16=True \
        models.policy_trainer.sequence_parallel=True \
        models.policy_trainer.use_distributed_optimizer=True \
        models.policy_trainer.recompute_granularity='full'  \
        models.policy_trainer.recompute_method='uniform' \
        models.policy_trainer.recompute_num_layers=1 \
        models.policy_trainer.recompute_modules="['mla_up_proj', 'core_attn', 'mlp']" \
        models.policy_trainer.tensor_model_parallel_size=4 \
        models.policy_trainer.pipeline_model_parallel_size=4 \
        models.policy_trainer.expert_tensor_parallel_size=1 \
        models.policy_trainer.expert_model_parallel_size=8 \
        models.policy_trainer.context_parallel_size=1 \
        models.policy_trainer.generation_batch_size=128 \
        models.policy_trainer.load=${mcore_ckpt_path} \
        models.policy_trainer.optimizer.lr=7e-7 \
        models.policy_trainer.optimizer.min_lr=7e-7 \
        models.policy_trainer.pos_clip_ratio=0.2 \
        models.policy_trainer.neg_clip_ratio=0.2 \
        models.reward.generation_batch_size=64 \
        models.policy.load=${hf_ckpt_path} \
        models.policy.generation_batch_size=128 \
        models.policy.tensor_model_parallel_size=32 \
        models.policy.max_prompt_tokens_length=768 \
        models.policy.max_response_tokens_length=2048 \
        models.policy.num_inference_per_prompt=32 \
        models.policy.gpu_memory_utilization=0.6 \
        models.policy.enable_thinking=False \
        2>&1 | tee ${log_file} ; exit ${PIPESTATUS[0]}