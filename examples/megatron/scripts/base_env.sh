#!/bin/bash

ray stop

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export OPENBLAS_NUM_THREADS=1


[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=8
[ -z "$RANK" ] && export RANK=0
if [ -z "${CUSTOM_PORTS}" ]; then
  set +x
  ports="30000"
  for i in $(seq 30001 30050); do
    ports="${ports};${i}"
  done
  set -x
  export CUSTOM_PORTS=$ports
  [ -z "$LOCAL_MASTER_ADDR" ] && export LOCAL_MASTER_ADDR=$MASTER_ADDR
  echo LOCAL_MASTER_ADDR=$MASTER_ADDR
fi

if [ -z "${MEGATRON}" ]; then
  echo "please set Megatron path"
  exit 1
fi
if [ -z "$CHATLEARN" ]; then
  echo "please set CHATLEARN path"
  exit 1
fi
if [ -z "$DATASET_PATH" ]; then
  echo "please set DATASET_PATH"
  exit 1
fi
if [ -z "model_size" ]; then
  echo "please set model_size"
  exit 1
fi

rm core*
rm ${MEGATRON}/megatron/fused_kernels/${build_path}/lock

export PYTHONPATH=${MEGATRON}:${CHATLEARN}:${PYTHONPATH}
export num_gpu=$(($WORLD_SIZE * $GPUS_PER_NODE))

[ -z "$num_gpu_policy" ] && export num_gpu_policy=$num_gpu
[ -z "$num_gpu_ref" ] && export num_gpu_ref=$num_gpu
[ -z "$num_gpu_reward" ] && export num_gpu_reward=$num_gpu
[ -z "$num_gpu_value" ] && export num_gpu_value=$num_gpu
[ -z "$num_gpu_ppo_policy" ] && export num_gpu_ppo_policy=$num_gpu
[ -z "$num_gpu_ppo_value" ] && export num_gpu_ppo_value=$num_gpu

if [[ "$model_size" == "gpt-345M" ]]; then
  export policy_num_layers=12
  export policy_hidden_size=512
  export policy_num_attention_heads=8
  export reward_num_layers=12
  export reward_hidden_size=512
  export reward_num_attention_heads=8
elif [[ "$model_size" == "gpt-30B" ]]; then
  export policy_num_layers=48
  export policy_hidden_size=7168
  export policy_num_attention_heads=56
  export reward_num_layers=48
  export reward_hidden_size=7168
  export reward_num_attention_heads=56
elif [[ "$model_size" == "gpt-13B" ]]; then
  export policy_num_layers=40
  export policy_hidden_size=5120
  export policy_num_attention_heads=40
  export reward_num_layers=40
  export reward_hidden_size=5120
  export reward_num_attention_heads=40
elif [[ "$model_size" == "gpt-7B" ]]; then
  export policy_num_layers=32
  export policy_hidden_size=4096
  export policy_num_attention_heads=32
  export reward_num_layers=32
  export reward_hidden_size=4096
  export reward_num_attention_heads=32
elif [[ "$model_size" == "gpt-66B" ]]; then
  export policy_num_layers=64
  export policy_hidden_size=9216
  export policy_num_attention_heads=72
  export reward_num_layers=64
  export reward_hidden_size=9216
  export reward_num_attention_heads=72
elif [[ "$model_size" == "gpt-175B" ]]; then
  export policy_num_layers=96
  export policy_hidden_size=12288
  export policy_num_attention_heads=96
  export reward_num_layers=96
  export reward_hidden_size=12288
  export reward_num_attention_heads=96
elif [[ "$model_size" == "llama2-7B" ]]; then
  export policy_num_layers=32
  export policy_hidden_size=4096
  export policy_num_attention_heads=32
  export policy_num_query_groups=32
  export policy_ffn_hidden_size=11008
  export reward_num_layers=32
  export reward_hidden_size=4096
  export reward_num_query_groups=32
  export reward_num_attention_heads=32
  export reward_ffn_hidden_size=11008
  export max_position_embedding=2048
elif [[ "$model_size" == "llama2-13B" ]]; then
  export policy_num_layers=40
  export policy_hidden_size=5120
  export policy_num_attention_heads=40
  export policy_ffn_hidden_size=13824
  export policy_num_query_groups=40
  export reward_num_layers=40
  export reward_hidden_size=5120
  export reward_num_attention_heads=40
  export reward_ffn_hidden_size=13824
  export reward_num_query_groups=40
elif [[ "$model_size" == "llama2-70B" ]]; then
  export policy_num_layers=80
  export policy_hidden_size=8192
  export policy_num_attention_heads=64
  export policy_ffn_hidden_size=28672
  export policy_num_query_groups=8
  export reward_num_layers=80
  export reward_hidden_size=8192
  export reward_num_attention_heads=64
  export reward_ffn_hidden_size=28672
  export reward_num_query_groups=8
  export group_query_attention=True
else
  echo "unsupported model_size ${model_size}, please set your own model config"
  exit 1
fi
