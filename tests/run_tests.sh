#!/bin/bash
export PYTHONPATH=$(cd ../ && pwd):${PWD}:${PYTHONPATH}
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
rm -rf core*
rm -rf /tmp/ray/*

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
if [ -d checkpoint ]; then
    rm -r checkpoint
fi
if [ -d checkpoint2 ]; then
    rm -r checkpoint2
fi

while getopts 'LM:C:V:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
      ;;
    V)
      VERBOSITY=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))


function run_test {
  attempts=0
  while [[ $attempts -lt 3 ]]; do
      rm -rf core*
      ray stop --force
      time "$@"
      if [[ $? -eq 0 ]]; then
          echo "$@ success"
          break
      fi

      attempts=$((attempts + 1))
      if [[ $attempts -lt 3 ]]; then
          echo "$file fail, retry ($attempts/3)... $@"
      else
          echo "$file fail, exit ... $@"
          exit 1
      fi
  done
}


function run_all_tests {
  run_test python train_o1.py -c configs/o1.yaml
  run_test python test_model_flow.py -c "configs/rlhf.yaml"
  run_test python test_data_dp.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_data_input.py -c "configs/exp.yaml"
  run_test python test_data_dp_zero.py -c "configs/rlhf.yaml"
  run_test python test_data_dp_ep.py -c "configs/rlhf.yaml"
  run_test python test_data_tp_ep.py -c "configs/rlhf.yaml"
  run_test python test_data_tp_ep_pp.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_colocate_forward_train.py -c "configs/rlhf2.yaml"
  run_test python test_evaluator_multi.py -c "configs/test_eval2.yaml"
  run_test python test_rlhf_cpu.py -c "configs/rlhf_cpu.yaml"
  run_test python test_placement_colocate4.py -c "configs/rlhf.yaml"
  run_test python test_evaluator2.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_custom.py -c "configs/rlhf.yaml"
  run_test python test_evaluator.py -c "configs/rlhf.yaml"
  run_test python test_evaluator3.py -c "configs/rlhf_eval.yaml"
  run_test python test_fixed_data.py -c "configs/rlhf.yaml"
  run_test python test_dynamic_data.py -c "configs/rlhf.yaml"
  run_test python test_relay_buffer.py -c "configs/rlhf.yaml"
  batch_size=5 run_test python test_relay_buffer.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_placement_colocate.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_placement_colocate2.py -c "configs/rlhf.yaml"
  run_test python test_placement_colocate3.py -c "configs/rlhf.yaml"
  run_test python test_placement_colocate2.py -c "configs/rlhf.yaml"
  RUN_FLAG=0 run_test python test_rlhf_ckpt.py -c "configs/rlhf.yaml"
  RUN_FLAG="resume" run_test python test_rlhf_ckpt.py -c "configs/rlhf.yaml"
  RUN_FLAG=0 run_test python test_rlhf_ckpt_replica.py -c "configs/rlhf.yaml"
  RUN_FLAG="resume" run_test python test_rlhf_ckpt_replica.py -c "configs/rlhf.yaml"
  run_test python test_timers.py
  run_test python test_rlhf_no_replica.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_replica2.py -c "configs/rlhf.yaml"
  run_test python test_send_recv.py
  run_test python test_rlhf_replica.py -c "configs/rlhf.yaml"
  run_test python test_rlhf.py -c "configs/rlhf.yaml"
  run_test python test_args.py -c "configs/exp.yaml"
  run_test python test_utils.py
  run_test python test_sampler.py
  run_test python test_distactor.py -c "configs/exp.yaml"
  run_test python test_placement.py -c "configs/exp.yaml"
  run_test python test_placement_colocate.py -c "configs/exp.yaml"
  run_test python test_flat_tensors.py
  run_test python test_balanced_tp.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_1.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_2.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_3.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_4.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_5.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_6.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_pp_unbalance_ep_tp_1.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_pp_unbalance_ep_tp_2.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_unbalance_ep_tp_pp_1.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_unbalance_ep_tp_pp_2.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_unbalance_ep_tp_pp_3.py -c "configs/test_param_sync.yaml"
  enable_indivisible_batch_size=True run_test python test_indivisible_batchsz.py -c "configs/rlhf.yaml"
}

if [ "$1" == "" ]; then
  if [ "$LOGFILE" != "" ]; then
    run_all_tests 2>&1 | tee $LOGFILE ; exit ${PIPESTATUS[0]}
  else
    run_all_tests
  fi
elif [ "$1" == "test_o1" ]; then
  run_test python train_o1.py -c configs/o1.yaml
elif [ "$1" == "test_data" ]; then
  run_test python test_data_dp.py -c "configs/rlhf.yaml"
  run_test python test_data_dp_zero.py -c "configs/rlhf.yaml"
  run_test python test_data_dp_ep.py -c "configs/rlhf.yaml"
  run_test python test_data_tp_ep.py -c "configs/rlhf.yaml"
  run_test python test_data_tp_ep_pp.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_unbalance_tp" ]; then
  run_test python test_unbalance_tp.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_1.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_2.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_3.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_4.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_5.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp_6.py -c "configs/test_param_sync.yaml"
elif [ "$1" == "test_hep" ]; then
  run_test python test_balance_hep_pp_unbalance_ep_tp_1.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_pp_unbalance_ep_tp_2.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_unbalance_ep_tp_pp_1.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_unbalance_ep_tp_pp_2.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_unbalance_ep_tp_pp_3.py -c "configs/test_param_sync.yaml"
elif [ "$1" == "test_validate_param_sync" ]; then
  run_test python test_balanced_tp.py -c "configs/test_param_sync.yaml"
  run_test python test_unbalance_tp.py -c "configs/test_param_sync.yaml"
  run_test python test_balance_hep_pp_unbalance_ep_tp_1.py -c "configs/test_param_sync.yaml"
elif [ "$1" == "test_fixed_data" ]; then
  run_test python test_fixed_data.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_dynamic_data" ]; then
  run_test python test_dynamic_data.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_relay_buffer" ]; then
  run_test python test_relay_buffer.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_placement_colocate" ]; then
  run_test python test_placement_colocate3.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_placement_colocate2.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_placement_colocate.py -c "configs/rlhf.yaml"
  run_test python test_placement_colocate2.py -c "configs/rlhf.yaml"
  run_test python test_placement_colocate.py -c "configs/exp.yaml"
elif [ "$1" == "test_rlhf_ckpt" ]; then
  RUN_FLAG=0 run_test python test_rlhf_ckpt.py -c "configs/rlhf.yaml"
  RUN_FLAG="resume" run_test python test_rlhf_ckpt.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_timers" ]; then
  run_test python test_timers.py
elif [ "$1" == "test_rlhf_no_replica" ]; then
  run_test python test_rlhf_no_replica.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_rlhf_replica" ]; then
  run_test python test_rlhf_replica2.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_replica.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_send_recv" ]; then
  run_test python test_send_recv.py
elif [ "$1" == "test_rlhf" ]; then
  run_test python test_rlhf.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_rlhf_custom" ]; then
  run_test python test_rlhf_custom.py -c "configs/rlhf.yaml"
elif [ "$1" == "test_args" ]; then
  run_test python test_args.py -c "configs/exp.yaml"
elif [ "$1" == "test_utils" ]; then
  run_test python test_utils.py
elif [ "$1" == "test_distactor" ]; then
  run_test python test_distactor.py -c "configs/exp.yaml"
elif [ "$1" == "test_placement" ]; then
  run_test python test_placement.py -c "configs/exp.yaml"
elif [ "$1" == "test_evaluator" ]; then
  run_test python test_evaluator.py -c "configs/rlhf.yaml"
  run_test python test_evaluator2.py -c "configs/rlhf.yaml"
  run_test python test_evaluator3.py -c "configs/rlhf_eval.yaml"
  run_test python test_evaluator_multi.py -c "configs/test_eval2.yaml"
elif [ "$1" == "test_indivisible_batchsz" ]; then
  enable_indivisible_batch_size=True run_test python test_indivisible_batchsz.py -c "configs/rlhf.yaml"
else
  echo -e "\033[31m$(date '+%Y-%m-%d %T.%N') [ERROR]: Unrecognized test name '$1'!\033[0m"
  exit -1
fi

ray stop --force
