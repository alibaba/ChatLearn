#!/bin/bash
set -x

export model_size=llama2-7B
export policy_tp=4
export reward_tp=4

cd ${CHATLEARN}/examples/megatron/aligment
source scripts/base_env.sh

if [ -z "${exp_name}" ]; then
  export exp_name=$(date +%F)-eval-${model_size}-${model_size}
fi

# megatron or vllm
backend=${1:-megatron}

scripts=tests/get_eval_reward.py
if [[ $backend == "vllm" ]];then
  export ENABLE_VLLM=True
  export PYTHONPATH=$PYTHONPATH:$CKPT_ROOT/$MODEL
  configs=configs/llama2/eval_vllm.yaml
else
  configs=configs/llama2/eval.yaml
fi

export eval_data_path=$DATA_ROOT/dev.jsonl

[ -z "$exp_name" ] && export exp_name=$(date +%F)-${model_size}-${trainer_engine}
[ -z "$output_dir" ] && export output_dir=${CHATLEARN}/output/
[ -z "$sample_per_episode" ] && sample_per_episode=1024
[ -z "$tokenizer_load" ] && export tokenizer_load=path-to-hf-tokenizer-for-vllm-backend
output_dir=${output_dir}/${exp_name}
mkdir -p ${output_dir}/
log_file=${output_dir}/log_${RANK}.log

policy_inference_load=${POLICY_LOAD} \
policy_load_iteration=${POLICY_LOAD_ITERATION} \
load_iteration=${REWARD_LOAD_ITERATION} \
reward_load_iteration=${REWARD_LOAD_ITERATION} \
reward_load=${REWARD_LOAD} \
tokenizer_model=${TOKENIZER_MODEL} \
num_gpu=${num_gpu} \
eval_data_path=${DATASET_PATH} \
python $scripts -c $configs 2>&1 | tee -a ${log_file} ; exit ${PIPESTATUS[0]}
