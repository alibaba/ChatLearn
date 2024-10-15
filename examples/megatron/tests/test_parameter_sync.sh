#!/bin/bash
set -x


[ -z "$model_size" ] && export model_size=gpt-345M

backend=${1:-megatron}
if [[ "$backend" != "megatron" ]] && [[ "$backend" != "vllm" ]]; then
  echo "ERROR: expect megatron or vllm backend, while "$backend
  exit 1
fi

source scripts/base_env.sh


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

[ -z "$policy_tp" ] && export policy_tp=8
[ -z "$ppo_policy_pp" ] && export ppo_policy_pp=1
[ -z "$reward_tp" ] && export reward_tp=8
[ -z "$ppo_value_pp" ] && export ppo_value_pp=1
[ -z "$policy_generation_batch_size" ] && export policy_generation_batch_size=32
export train_micro_batch_size=2

config_dir=${CHATLEARN}/examples/megatron/configs/

if [[ $model_size == "gpt"* ]]; then
  if [[ "$backend" != "megatron" ]]; then
    echo "ERROR: gpt model support megatron backend for now."
    exit 1
  fi
  config_name=${config_dir}/gpt/rlhf.yaml
  export vocab_file=${DATA_DIR}/gpt2-vocab.json
  export merge_file=${DATA_DIR}/gpt2-merges.txt
elif [[ $model_size == "llama2"* ]]; then
  if [[ "$backend" == "megatron" ]]; then
    config_name=${config_dir}/llama2/rlhf.yaml
  else
    export ENABLE_VLLM=True
    config_name=${config_dir}/llama2/vllm_rlhf.yaml
  fi
  export tokenizer_model=$VOCAB_FILE/tokenizer.model
  export tokenizer_load=$VOCAB_FILE
else
  echo "unexpected model_type $model_size."
  exit 1
fi

num_episode=2 \
data_path=${DATASET_PATH} \
enable_lora_value=${lora} \
enable_lora_policy=${lora} \
tensorboard_dir=${TENSORBOARD_DIR} \
python tests/test_parameter_sync.py -c ${config_name} 2>&1 | tee ${output_dir}/log_${RANK}.log ; exit ${PIPESTATUS[0]}
