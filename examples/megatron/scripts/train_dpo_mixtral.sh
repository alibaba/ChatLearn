#!/bin/bash
set -x

[ -z "$model_size" ] && export model_size=mixtral-8x7B

# Get the directory of the current script
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source ${DIR}/base_env.sh

export trainer_engine=dpo

# clip
export clip_grad=5.0

# desable dropout
export attention_dropout=0.0
export hidden_dropout=0.0
export retro_encoder_hidden_dropout=0.0
export retro_encoder_attention_dropout=0.0


if [[ "$model_size" == "mixtral-8x7B" ]]; then
    export policy_tp=1
    export policy_ep=8
    export ppo_policy_pp=4
    export ref_pp=4
    export train_global_batch_size=128
    export ref_generation_batch_size=2
    export train_micro_batch_size=1
    export policy_recompute_activations=True
    export policy_moe_layer_recompute=True
fi

configs=$CHATLEARN/examples/megatron/configs/mixtral/dpo.yaml

[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$sample_per_episode" ] && sample_per_episode=1024

output_dir=${output_dir}/${exp_name}
export data_checkpoint_path=${output_dir}/data_checkpoint
mkdir -p $output_dir
log_file=${output_dir}/log_${RANK}.log

policy_inference_load=${POLICY_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_gpu=${num_gpu} \
data_path=${DATASET_PATH} \
sample_per_episode=${sample_per_episode} \
python entry/train_dpo.py -c $configs 2>&1 | tee -a ${log_file} ; exit ${PIPESTATUS[0]}


