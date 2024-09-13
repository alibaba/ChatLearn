#!/bin/bash
set -x


[ -z "$model_size" ] && export model_size=gpt-7B

# Get the directory of the current script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ${DIR}/base_env.sh


[ -z "$max_new_tokens" ] && export max_new_tokens=512
[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$DATA_DIR" ] && DATA_DIR=${output_dir}/gpt/
output_dir=${output_dir}/${exp_name}
export data_checkpoint_path=${output_dir}/data_checkpoint

mkdir -p $output_dir

if [[ ! -f "${DATA_DIR}/gpt2-vocab.json" ]]; then
  wget -P $DATA_DIR https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
  wget -P $DATA_DIR https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi

export max_seq_len=$(( max_new_tokens*2 ))
export lora=False

# parallel strategy and batch size, please adjust them accordingly
if [[ "$model_size"  == "gpt-7B" ]]; then
  [ -z "$policy_tp" ] && export policy_tp=8
  [ -z "$ppo_policy_pp" ] && export ppo_policy_pp=1
  [ -z "$reward_tp" ] && export reward_tp=8
  [ -z "$ppo_value_pp" ] && export ppo_value_pp=1
  export batch_generation_min_prompt_length=32
  export num_gpu_ref=8
  export num_gpu_value=8
  [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=256
  [ -z "$ref_generation_bs" ] && export ref_generation_bs=32
  [ -z "$value_generation_bs" ] && export value_generation_bs=32
  [ -z "$reward_generation_bs" ] && export reward_generation_bs=32
  [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=8
  [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
elif [[ "$model_size"  == "gpt-13B" ]]; then
  [ -z "$policy_tp" ] && export policy_tp=8
  [ -z "$ppo_policy_pp" ] && export ppo_policy_pp=2
  [ -z "$reward_tp" ] && export reward_tp=8
  [ -z "$ppo_value_pp" ] && export ppo_value_pp=2
  export batch_generation_min_prompt_length=32
  export num_gpu_ref=8
  export num_gpu_value=8
  [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=180
  [ -z "$ref_generation_bs" ] && export ref_generation_bs=64
  [ -z "$value_generation_bs" ] && export value_generation_bs=64
  [ -z "$reward_generation_bs" ] && export reward_generation_bs=64
  [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=8
  [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
elif [[ "$model_size"  == "gpt-30B" ]]; then
  [ -z "$policy_tp" ] && export policy_tp=8
  [ -z "$ppo_policy_pp" ] && export ppo_policy_pp=4
  [ -z "$reward_tp" ] && export reward_tp=8
  [ -z "$ppo_value_pp" ] && export ppo_value_pp=4
    export num_gpu_ref=16
    export num_gpu_value=16
    export ref_pp=2
    export reward_pp=2
    export batch_generation_min_prompt_length=32
    export free_memory_ppo_value=True
    export free_memory_ppo_policy=True
    [ -z "$inference_batch_times_seqlen_threshold" ] && export inference_batch_times_seqlen_threshold=4096
    [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=64
    [ -z "$ref_generation_bs" ] && export ref_generation_bs=64
    [ -z "$value_generation_bs" ] && export value_generation_bs=32
    [ -z "$reward_generation_bs" ] && export reward_generation_bs=64
    [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=4
    [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
elif [[ "$model_size"  == "gpt-66B" ]]; then
    export lora=True
    [ -z "$policy_tp" ] && export policy_tp=8
    [ -z "$ppo_policy_pp" ] && export ppo_policy_pp=4
    [ -z "$reward_tp" ] && export reward_tp=8
    [ -z "$ppo_value_pp" ] && export ppo_value_pp=4
    export policy_recompute_activations=True
    export policy_recompute_granularity=full
    export value_recompute_activations=True
    export value_recompute_granularity=full
    export free_memory_ppo_value=True
    export free_memory_ppo_policy=True
    export batch_generation_min_prompt_length=32
    export num_gpu_ref=16
    export num_gpu_reward=32
    export num_gpu_value=16
    export ref_pp=2
    export reward_pp=2
    export inference_batch_times_seqlen_threshold=16384
    [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=64
    [ -z "$ref_generation_bs" ] && export ref_generation_bs=64
    [ -z "$value_generation_bs" ] && export value_generation_bs=32
    [ -z "$reward_generation_bs" ] && export reward_generation_bs=64
    [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=4
    [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
elif [[ "$model_size"  == "gpt-175B" ]]; then
    [ -z "$policy_tp" ] && export policy_tp=16
    [ -z "$ppo_policy_pp" ] && export ppo_policy_pp=8
    [ -z "$reward_tp" ] && export reward_tp=16
    [ -z "$ppo_value_pp" ] && export ppo_value_pp=8
    export free_memory_ppo_value=True
    export free_memory_ppo_policy=True
    export policy_recompute_activations=True
    export policy_recompute_granularity=full
    export value_recompute_activations=True
    export value_recompute_granularity=full
    export batch_generation_min_prompt_length=32
    export num_gpu_ref=64
    export num_gpu_reward=128
    export num_gpu_value=64
    export ref_pp=4
    export reward_pp=4
    export value_pp=4
    export inference_batch_times_seqlen_threshold=4096
    [ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=64
    [ -z "$ref_generation_bs" ] && export ref_generation_bs=64
    [ -z "$value_generation_bs" ] && export value_generation_bs=64
    [ -z "$reward_generation_bs" ] && export reward_generation_bs=64
    [ -z "$train_micro_batch_size" ] && export train_micro_batch_size=2
    [ -z "$train_global_batch_size" ] && export train_global_batch_size=512
fi
config_dir=${CHATLEARN}/examples/megatron/configs/

data_path=${DATASET_PATH} \
vocab_file=${DATA_DIR}/gpt2-vocab.json \
merge_file=${DATA_DIR}/gpt2-merges.txt \
enable_lora_value=${lora} \
enable_lora_policy=${lora} \
tensorboard_dir=${TENSORBOARD_DIR} \
python entry/train_rlhf.py -c ${config_dir}/gpt/rlhf.yaml 2>&1 | tee ${output_dir}/log_${RANK}.log ; exit ${PIPESTATUS[0]}