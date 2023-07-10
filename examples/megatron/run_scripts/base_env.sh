ray stop

export NCCL_SOCKET_IFNAME=eth0
export RANK=${1:-0}
export LOCAL_MASTER_ADDR=localhost
export CUDA_DEVICE_MAX_CONNECTIONS=1

ports="9000"
for i in $(seq 9001 9050); do
  ports="${ports};${i}"
done
export CUSTOM_PORTS=$ports

export ROOT=/mnt/shared/Group-m6/tianhang_zhu/latest_rlhf_0609
export ROOT=/mnt/user/E-xianyan.xianyanjia-189885/
export rlhf=${ROOT}/rlhf
export Megatron=${ROOT}/QWen2

rm core*
rm ${Megatron}/megatron/fused_kernels/build/lock

export PYTHONPATH=${Megatron}/megatron_rlhf:${rlhf}:${Megatron}:${PYTHONPATH}
