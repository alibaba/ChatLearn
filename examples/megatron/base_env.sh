export NCCL_SOCKET_IFNAME=eth0
MASTER_ADDR=localhost
export LOCAL_MASTER_ADDR=$MASTER_ADDR

export CUDA_DEVICE_MAX_CONNECTIONS=1
export RANK=0
ray stop
pip install protobuf==3.19.6
ports="9000"
for i in $(seq 9001 9050); do
  ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports
ROOT=/mnt/user/E-xianyan.xianyanjia-189885/
rlhf=${ROOT}/rlhf-llama
Megatron=${ROOT}/Megatron-LM-llama

rm core*
rm ${Megatron}/megatron/fused_kernels/build/lock

export PYTHONPATH=${Megatron}/megatron_rlhf:${rlhf}:${Megatron}:${PYTHONPATH}

cd ${rlhf}/examples/megatron
