#!/bin/bash
set -exo pipefail
export PYTHONPATH=$(cd ../ && pwd):${PYTHONPATH}
CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
LOGFILE=/tmp/pytorch_py_test.log
rm -rf core*
MAX_GRAPH_SIZE=500
GRAPH_CHECK_FREQUENCY=100
VERBOSITY=2

while getopts 'LM:C:V:' OPTION
do
  case $OPTION in
    L)
      LOGFILE=
      ;;
    M)
      MAX_GRAPH_SIZE=$OPTARG
      ;;
    C)
      GRAPH_CHECK_FREQUENCY=$OPTARG
      ;;
    V)
      VERBOSITY=$OPTARG
      ;;
  esac
done
shift $(($OPTIND - 1))



function run_test {
  "$@"
}


function run_all_tests {
  run_test python test_fixed_data.py -c "configs/rlhf.yaml"
  run_test python test_dynamic_data.py -c "configs/rlhf.yaml"
  run_test python test_relay_buffer.py -c "configs/rlhf.yaml"
  run_test python test_placement_colocate3.py -c "configs/rlhf.yaml"
  run_test python test_placement_colocate2.py -c "configs/rlhf.yaml"
  RUN_FLAG=0 run_test python test_rlhf_ckpt.py -c "configs/rlhf.yaml"
  RUN_FLAG="resume" run_test python test_rlhf_ckpt.py -c "configs/rlhf.yaml"
  run_test python test_timers.py
  run_test python test_rlhf_no_replica.py -c "configs/rlhf.yaml"
  run_test python test_rlhf_replica2.py -c "configs/rlhf.yaml"
  run_test python test_send_recv.py
  run_test python test_rlhf_replica.py -c "configs/rlhf.yaml"
  run_test python test_rlhf.py -c "configs/rlhf.yaml"
  run_test python test_args.py -c "configs/exp.yaml"
  run_test python test_utils.py
  run_test python test_distactor.py -c "configs/exp.yaml"
  run_test python test_placement.py -c "configs/exp.yaml"
  run_test python test_placement_colocate.py -c "configs/exp.yaml"
}

if [ "$LOGFILE" != "" ]; then
  run_all_tests 2>&1 | tee $LOGFILE
else
  run_all_tests
fi
