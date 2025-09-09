#!/bin/bash

# Tested on 8xH20-3e with 140G VRAM
set -x
# pip install "sglang[all]>=0.5.1.post2"
export CHATLEARN=$(pwd)
export PYTHONPATH=${CHATLEARN}:${PYTHONPATH}
source scripts/base_env.sh
export RAY_DEDUP_LOGS=1

export exp_name=debug_mini
python chatlearn/entrypoint.py grpo \
        --config-file template/grpo_fsdp_agent.yaml \
        runtime_args.exp_name=${exp_name} \
        runtime_args.rollout_backend=sglang \
        runtime_args.data_path=${CHATLEARN}/dataset/MATH-lighteval/train_agent.json \
        runtime_args.eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test_agent.json \
        runtime_args.output_dir=${CHATLEARN}/output/${exp_name} \
        runtime_args.num_episode=200 \
        runtime_args.sample_per_episode=128 \
        runtime_args.train_global_batch_size=128 \
        runtime_args.train_micro_batch_size=64 \
        runtime_args.save_episode_interval=200 \
        runtime_args.eval_episode_interval=5 \
        runtime_args.enable_eval_before_training=False \
        runtime_args.log_args_dict.enable_wandb=False \
        runtime_args.log_args_dict.wandb_project=your_wandb_project \
        models.policy_trainer.num_gpu=2 \
        models.policy_trainer.packing=True \
        models.policy_trainer.meta_init=False \
        models.policy_trainer.groupgemm=False \
        models.policy_trainer.generation_batch_size=64 \
        models.policy_trainer.ulysses_sequence_parallel_size=1 \
        models.policy_trainer.load=${CHATLEARN}/Qwen3-1.7B/ \
        models.policy_trainer.optimizer.lr=2e-6 \
        models.policy_trainer.pos_clip_ratio=0.2 \
        models.policy_trainer.neg_clip_ratio=0.2 \
        models.ref_policy.generation_batch_size=64 \
        models.policy.is_sync_mode=False \
        models.policy.generation_batch_size=64 \
        models.policy.enforce_eager=False \
        models.policy.tensor_model_parallel_size=1 \
        models.policy.max_prompt_tokens_length=1024 \
        models.policy.max_response_tokens_length=2048 \
        models.policy.num_inference_per_prompt=32 \
        models.policy.gpu_memory_utilization=0.85 \
        models.policy.enable_thinking=False \
        models.reward.generation_batch_size=256 \
        2>&1 | tee log_${exp_name}.log ; exit ${PIPESTATUS[0]}