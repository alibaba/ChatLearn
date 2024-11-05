#!/bin/bash
set -x

[ -z "$MEGATRON" ] && export MEGATRON=path-to-megatron
[ -z "$CHATLEARN" ] && export CHATLEARN=path-to-chatlearn
[ -z "$TP" ] && export TP=4
[ -z "$PP" ] && export PP=2
[ -z "$VOCAB_FILE" ] && export VOCAB_FILE=path-to-tokenizer
[ -z "$LOAD" ] && export LOAD=path-to-ckpt
[ -z "$DATASET_PATH" ] && export DATASET_PATH=path-to-dataset-json
[ -z "$model_size" ] && export model_size=llama2-13B
[ -z "$tokenizer_load" ] && export tokenizer_load=path-to-hf-tokenizer-for-vllm-backend

cd $CHATLEARN/examples/megatron

# megatron or vllm
backend=${1:-megatron}

if [[ "$backend" != "megatron" ]] && [[ "$backend" != "vllm" ]]; then
  echo "ERROR: expect megatron or vllm backend, while "$backend
  exit 1
fi

if [[ $model_size == "gpt"* ]]; then
  if [[ "$backend" != "megatron" ]]; then
    echo "ERROR: gpt model support megatron backend for now."
    exit 1
  fi
  configs=configs/gpt/test_policy.yaml
  export vocab_file=$VOCAB_FILE
  export merge_file=$MERGE_FILE
  export max_new_tokens=512
  export max_seq_len=1024
elif [[ $model_size == "llama2"* ]]; then
  if [[ "$backend" == "megatron" ]]; then
    configs=configs/llama2/test_policy.yaml
  else
    export ENABLE_VLLM=True
    configs=configs/llama2/test_vllm_policy.yaml
  fi
  export tokenizer_model=$VOCAB_FILE
else
  echo "unexpected model_type $model_size."
  exit 1
fi

source scripts/base_env.sh

[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/

output_dir=${output_dir}/${exp_name}
mkdir -p ${output_dir}/
log_file=${output_dir}/log_${RANK}.log

export batch_generation_min_prompt_length=32

generation_batch_size=64 \
num_gpu=${num_gpu:-8} \
policy_tp=$TP \
policy_pp=$PP \
eval_data_path=$DATASET_PATH \
policy_inference_load=$LOAD \
python tests/test_policy_generation.py -c $configs 2>&1 | tee ${log_file}.log ; exit ${PIPESTATUS[0]}
