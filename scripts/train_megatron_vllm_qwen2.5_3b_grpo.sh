#!/bin/bash
set -x

export CHATLEARN=$(pwd)
export MEGATRON_PATH=${CHATLEARN}/Megatron-LM
export PYTHONPATH=${CHATLEARN}:${MEGATRON_PATH}:${PYTHONPATH}
source scripts/base_env.sh


python chatlearn/entrypoint.py grpo --config-file template/grpo_megatron.yaml \
        runtime_args.exp_name=grpo_megatron \
        runtime_args.data_path=${CHATLEARN}/dataset/MATH-lighteval/train.json \
        runtime_args.eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test.json \
        runtime_args.output_dir=${CHATLEARN}/output/grpo_megatron \
        runtime_args.num_episode=200 \
        runtime_args.sample_per_episode=512 \
        runtime_args.train_global_batch_size=512 \
        runtime_args.train_micro_batch_size=8 \
        runtime_args.save_episode_interval=5 \
        runtime_args.eval_episode_interval=5 \
        runtime_args.enable_eval_before_training=false \
        runtime_args.log_args_dict.enable_wandb=False \
        runtime_args.log_args_dict.wandb_project=your_wandb_project \
        models.policy_trainer.num_gpu=${num_device} \
        models.policy_trainer.tensor_model_parallel_size=2 \
        models.policy_trainer.generation_batch_size=8 \
        models.policy_trainer.load=${CHATLEARN}/Qwen2.5-3B-Instruct-hf-to-mcore-tp2-pp1 \
        models.policy_trainer.optimizer.lr=2e-6 \
        models.policy_trainer.pos_clip_ratio=0.2 \
        models.policy_trainer.neg_clip_ratio=0.2 \
        models.ref_policy.tensor_model_parallel_size=2 \
        models.ref_policy.generation_batch_size=2 \
        models.policy.load=${CHATLEARN}/Qwen2.5-3B-Instruct \
        models.policy.generation_batch_size=256 \
        models.policy.tensor_model_parallel_size=2 \
        models.policy.seq_length=2048 \
        models.policy.max_seq_len_to_capture=2348 \
        models.policy.num_inference_per_prompt=32 \
        models.policy.gpu_memory_utilization=0.85 \
        models.policy.enable_thinking=False \
        models.reward.generation_batch_size=256 \
        2>&1 | tee log_grpo_megatron.log ; exit ${PIPESTATUS[0]}