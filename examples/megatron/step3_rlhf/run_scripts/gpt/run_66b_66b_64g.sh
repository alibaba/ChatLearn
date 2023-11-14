#!/bin/bash

set -x


export model_size=66B
source run_scripts/gpt/base_env.sh

export max_new_tokens=${1}
export lora=${2}
export max_seq_len=$(( max_new_tokens*2 ))


[ -z "$policy_tp" ] && export policy_tp=8
[ -z "$ppo_policy_pp" ] && export ppo_policy_pp=8
[ -z "$reward_tp" ] && export reward_tp=8
[ -z "$ppo_value_pp" ] && export ppo_value_pp=8
if [[ "$lora" == "True" ]]; then
    if [[ "$max_new_tokens" == "512" ]]; then
        export policy_recompute_activations=True
        export policy_recompute_granularity=full
        export value_recompute_activations=True
        export value_recompute_granularity=full
        export batch_generation_min_prompt_length=32
        export num_device_ref=32
        export num_device_reward=64
        export num_device_value=32
        export ref_pp=4
        export reward_pp=8
        export value_pp=4
        export inference_batch_times_seqlen_threshold=4096
        [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=128
        [ -z "$ref_generation_bs" ] && export ref_generation_bs=128
        [ -z "$value_generation_bs" ] && export value_generation_bs=128
        [ -z "$reward_generation_bs" ] && export reward_generation_bs=128
        [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=4
        [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
    elif [[ "$max_new_tokens" == "1024" ]]; then
        export policy_recompute_activations=True
        export policy_recompute_granularity=full
        export value_recompute_activations=True
        export value_recompute_granularity=full
        export batch_generation_min_prompt_length=32
        export num_device_ref=32
        export num_device_reward=64
        export num_device_value=32
        export ref_pp=4
        export reward_pp=8
        export value_pp=4
        export inference_batch_times_seqlen_threshold=4096
        [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=64
        [ -z "$ref_generation_bs" ] && export ref_generation_bs=64
        [ -z "$value_generation_bs" ] && export value_generation_bs=64
        [ -z "$reward_generation_bs" ] && export reward_generation_bs=64
        [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=2
        [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
    fi
fi

bash run_scripts/gpt/benchmark.sh

