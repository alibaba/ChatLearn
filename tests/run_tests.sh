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


shift $(($OPTIND - 1))


function run_test {
  rm -rf core*
  ray stop --force
  time "$@"
}

set -x

TEST_CASES=(
  "unittest"                   # passed
  # "base"                       # to be fixed
  # "rlhf"                       # to be fixed
  # "parameter_sync"             # to be fixed
  # "eval"                       # to be fixed
  # "o1"                         # to be fixed
  # "sprl"                       # to be fixed
)
# Run ALL Tests in TEST_CASES
pip install --no-cache-dir hydra-core==1.3.2
for test_case in "${TEST_CASES[@]}"
do
    run_test python test_main.py -t "$test_case" -c "configs/$test_case.yaml" || exit 1
done

# Usage: Run A Specified TestCase with case name
#run_test python test_main.py -t "rlhf.test_rlhf_ckpt" -c "configs/rlhf.yaml"
run_test python test_main.py -t "parameter_sync/v2" -c "configs/parameter_sync.yaml"
ray stop --force
