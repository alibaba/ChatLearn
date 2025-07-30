#!/bin/bash

ray stop

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN


[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU:-$(python -c "import torch; print(torch.cuda.device_count())")}
[ -z "$RANK" ] && export RANK=0
[ -z "$MASTER_PORT" ] && export MASTER_PORT=12456
[ -z "$NNODES" ] && export NNODES=${WORLD_SIZE:-1}
[ -z "$NODE_RANK" ] && export NODE_RANK=${RANK:-0}
if [ -z "${CUSTOM_PORTS}" ]; then
  set +x
  ports="30010"
  for i in $(seq 30011 30050); do
    ports="${ports};${i}"
  done
  set -x
  export CUSTOM_PORTS=$ports
  [ -z "$LOCAL_MASTER_ADDR" ] && export LOCAL_MASTER_ADDR=$MASTER_ADDR
  echo LOCAL_MASTER_ADDR=$MASTER_ADDR
fi

if [ -z "$CHATLEARN" ]; then
  echo "please set CHATLEARN path"
  exit 1
fi

rm core*

export num_device=$(($WORLD_SIZE * $GPUS_PER_NODE))