ROOT=/mnt/user/E-xianyan.xianyanjia-189885/
rlhf=${ROOT}/rlhf/
Megatron=${ROOT}/QWen/
export PYTHONPATH=${PYTHONPATH}:${Megatron}/megatron_rlhf:${rlhf}:${Megatron}

echo $PYTHONPATH
rm ${Megatron}/megatron/fused_kernels/build/lock
rm core*
ray stop
export NCCL_DEBUG=WARN
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=eth0
# export RANK=${1:-0}
# export LOCAL_MASTER_ADDR=localhost
#export LOCAL_MASTER_ADDR=172.24.122.48
# ports="9000"
# for i in $(seq 9001 9050); do
#        ports="${ports};${i}"
# done
# echo $ports
# export CUSTOM_PORTS=$ports
debug=0
if [[ "$debug" == "1" ]]; then
  sample_per_episode=384
else
  sample_per_episode=1152
fi
enable_nsys=0
label=tp4_test_reward_tp2_new_dynamic
label=13b_test_ref_cache2_2
label=profile_policy_generate_bladnn4
if [[ "$enable_nsys" == "1" ]]; then
  nsys_param="nsys profile -w true -t cuda,nvtx -s none  --capture-range=cudaProfilerApi --capture-range-end=stop -o ${label} --force-overwrite true "
else
  nsys_param=""
fi
#export RAY_memory_monitor_refresh_ms=0

build_path=build \
  data_path=/mnt/user/E-xianyan.xianyanjia-189885/data/v10.15.5-fix-seed.jsonl \
  sample_per_episode=$sample_per_episode \
  generation_batch_size=48 \
  ref_generation_bs=12 \
  train_micro_batch_size=8 \
  train_global_batch_size=64 \
  policy_generation_batch_size=96 \
  $nsys_param \
  python train_rlhf.py -c run_configs/configs_13b_350m/rlhf.yaml #> logs/$(date +%F)_${label}.log 2>&1&
