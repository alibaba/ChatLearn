#!/bin/bash
set -x

export CHATLEARN=$(pwd)
source scripts/base_env.sh

python chatlearn/entrypoint.py grpo \
        --config-file grpo.yaml \
        runtime_args.exp_name=grpo_fsdp \
        runtime_args.data_path=${CHATLEARN}/dataset/MATH-lighteval/train.json \
        runtime_args.eval_data_path=${CHATLEARN}/dataset/MATH-lighteval/test.json \
        runtime_args.output_dir=${CHATLEARN}/output/grpo_fsdp \
        runtime_args.num_episode=200 \
        runtime_args.sample_per_episode=512 \
        runtime_args.train_global_batch_size=512 \
        runtime_args.train_micro_batch_size=8 \
        runtime_args.save_episode_interval=5 \
        runtime_args.eval_episode_interval=5 \
        runtime_args.enable_eval_before_training=true \
        +models.policy_trainer.num_gpu=${num_device} \
        +models.policy_trainer.generation_batch_size=8 \
        +models.policy_trainer.load=${CHATLEARN}/Qwen3-4B \
        +models.policy_trainer.optimizer.lr=2e-6 \
        +models.policy_trainer.pos_clip_ratio=0.2 \
        +models.policy_trainer.neg_clip_ratio=0.2 \
        models.policy.generation_batch_size=256 \
        models.policy.num_gpu=${num_device} \
        models.policy.tensor_model_parallel_size=1 \
        models.policy.seq_length=2048 \
2>&1 | tee ${log_file}.log ; exit ${PIPESTATUS[0]}