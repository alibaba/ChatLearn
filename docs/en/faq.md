# Common Issues
## ECC Error
ECC Error is a machine failure. It is recommended to use [Continued Training and Fault Tolerance](tutorial/continue_train.md) to automatically blacklist faulty machines and restart the job.
## How to build a custom training flow for multiple reward models
The provided examples are for training a single reward model. If you need to customize the training flow for multiple reward models, please refer to [Custom Inference and Training Workflow](tutorial/custom_model_flow.md).
## RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding
```
RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding:
   size mismatch for weight: copying a param with shape torch.Size([xxx, xxx]) from checkpoint, the shape in the current model is torch.Size([[xxx, xxx]]).
```
This is generally caused by changes in the TP and requires adjusting the parameter `make_vocab_size_divisible_by` to align the shape of the padded embedding parameters.
## YAML Configuration
Refer to [Configuration File](config_yaml.md).
## How to enable 'Efficient memory sharing' to reduce memory usage
Refer to the documentation on [Efficient memory sharing](tutorial/ems.md).
## Megatron Model Conversion and Parallel Strategy
```bash
cd $CHATLEARN
model_type=GPT # for reward model, set model_type to REWARD
load_dir=xxx
save_dir=xxx
target_tp=xxx
target_pp=xxx
python chatlearn/tools/megatron_checkpoint_utils.py --model-type ${model_type} --load-dir ${load_dir} --save-dir ${save_dir} \
    --target-tensor-parallel-size ${target_tp} --target-pipeline-parallel-size ${target_pp}
```
Note that this script has only been validated on official Megatron-LM scripts.
## Applying for custom_port
In the DLC environment, the current RLHF training has already allocated 50 ports to meet all usage scenarios. It is recommended to set the advanced configuration as follows:
```
customPortList=30000-30050
```
## Task failure but DLC status shows success
1. Redirect the log to a file
```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt
```
In this situation, the exit code is always 0, and the DLC job will show as successful. It is necessary to change it to the following:
```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt ; exit ${PIPESTATUS[0]}
```
2. There are some additional operations after the training command, causing the error code to be different from the training command's error code. It is recommended to add `set -e` at the beginning of the command, so that it exits at the first encountered error command.
## Adjusting lr error in continued training
Megatron checks if the lr has changed during load_checkpoint. It is necessary to set the Megatron model parameter `override_opt_param_scheduler` to True to bypass the check.
## How to specify the frequency of model saving during training
In rlhf.yaml, configure `save_episode_interval`.