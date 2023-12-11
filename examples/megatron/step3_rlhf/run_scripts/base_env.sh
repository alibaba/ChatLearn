ray stop

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=8
[ -z "$RANK" ] && export RANK=0
if [ -z "${CUSTOM_PORTS}" ]; then
  ports="30000"
  for i in $(seq 30001 30050); do
    ports="${ports};${i}"
  done
  export CUSTOM_PORTS=$ports
  [ -z "$LOCAL_MASTER_ADDR" ] && export LOCAL_MASTER_ADDR=$MASTER_ADDR
  echo LOCAL_MASTER_ADDR=$MASTER_ADDR
fi


if [ -z "${MEGATRON}" ]; then
  echo "please set Megatron path"
fi
if [ -z "$CHATLEARN" ]; then
   echo "please set CHATLEARN path"
fi
if [ -z "$DATASET_PATH" ]; then
   echo "please set DATASET_PATH"
fi


rm core*
rm ${MEGATRON}/megatron/fused_kernels/${build_path}/lock

export PYTHONPATH=${MEGATRON}:${CHATLEARN}:${CHATLEARN}/examples/megatron:${PYTHONPATH}

echo set PYTHONPATH $PYTHONPATH

cd ${CHATLEARN}/examples/megatron/step3_rlhf

export num_device=$(($WORLD_SIZE * $GPUS_PER_NODE))
