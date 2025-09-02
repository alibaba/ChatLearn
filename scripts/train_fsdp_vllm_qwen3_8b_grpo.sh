#!/bin/bash

# Tested on 8xH20-3e with 140G VRAM
set -x

export CHATLEARN=$(pwd)
export PYTHONPATH=${CHATLEARN}:${PYTHONPATH}
source scripts/base_env.sh
export RAY_DEDUP_LOGS=1

export exp_name=qwen3-grpo-8b
python chatlearn/entrypoint.py grpo \
        --config-file template/grpo_fsdp.yaml \
        runtime_args.exp_name=${exp_name} \
        runtime_args.data_path=${CHATLEARN}/dataset/MATH-lighteval/train.json \
        runtime_args.eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test.json \
        runtime_args.output_dir=${CHATLEARN}/output/${exp_name} \
        runtime_args.num_episode=200 \
        runtime_args.sample_per_episode=512 \
        runtime_args.train_global_batch_size=512 \
        runtime_args.train_micro_batch_size=64 \
        runtime_args.save_episode_interval=5 \
        runtime_args.eval_episode_interval=5 \
        runtime_args.enable_eval_before_training=True \
        runtime_args.log_args_dict.enable_wandb=False \
        runtime_args.log_args_dict.wandb_project=your_wandb_project \
        models.policy_trainer.num_gpu=${num_device} \
        models.policy_trainer.packing=True \
        models.policy_trainer.meta_init=False \
        models.policy_trainer.groupgemm=False \
        models.policy_trainer.generation_batch_size=64 \
        models.policy_trainer.ulysses_sequence_parallel_size=1 \
        models.policy_trainer.load=${CHATLEARN}/pretrained_models/Qwen3-8B/ \
        models.policy_trainer.optimizer.lr=2e-6 \
        models.policy_trainer.pos_clip_ratio=0.2 \
        models.policy_trainer.neg_clip_ratio=0.2 \
        models.ref_policy.generation_batch_size=64 \
        models.policy.generation_batch_size=64 \
        models.policy.enforce_eager=False \
        models.policy.tensor_model_parallel_size=1 \
        models.policy.seq_length=2048 \
        models.policy.max_seq_len_to_capture=2348 \
        models.policy.num_inference_per_prompt=32 \
        models.policy.gpu_memory_utilization=0.85 \
        models.policy.enable_thinking=False \
        models.reward.generation_batch_size=256 \
        2>&1 | tee log_${exp_name}.log ; exit ${PIPESTATUS[0]}