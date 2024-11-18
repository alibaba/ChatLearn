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
## Failure when converting checkpoint
Using Megatron-LM version core_r0.8.0 as the backend to convert checkpoints may cause the following error:

```bash
...
File "/root/Megatron-LM/megatron/training/checkpointing.py", line 426, in save_checkpoint
    logger.debug(f"rank: {torch.distributed.get_rank()}, takes {end_misc - start_misc} to finalize ckpt save ")
File "/usr/local/lib/python3.10/dist-packages/torch/distributed/distributed_c10d.py", line 1779, in get_rank
    default_pg = _get_default_group()
File "/usr/local/lib/python3.10/dist-packages/torch/distributed/distributed_c10d.py", line 1001, in _get_default_group
    raise ValueError(
 ValueError: Default process group has not been initialized, please make sure to call init_process_group.
```

This issue arises due to the lack of initialization of the default process group when converting checkpoints. It is introduced in Megatron-LM version core_r0.8.0. There are two possible solutions to address this problem:


1. Consider commenting out the problematic line because it only affects the debug-level logging output.

2. Alternatively, consider using Megatron-LM version core_r0.9.0 as the backend, as the bug has been fixed in this version. However, the correctness and performance of this version have not been validated for ChatLearn yet. We plan to upgrade our supported version of Megatron-LM to core_r0.9.0 in the future.

## Alignment training with pipeline parallelism may encounter non-contiguous tensors

If you are using Megatron-LM as the backend for alignment training and enable pipeline parallelism, you may encounter the following issue:

```bash
Traceback (most recent call last):
  File "/root/ChatLearn/chatlearn/runtime/decorator.py", line 166, in inner
    return func(self, *args, **kwargs)
    ret = func(self, *args, **kwargs)
  File "/root/ChatLearn/examples/megatron/models/old_policy_inference.py", line 408, in forward_step
    return self._forward_step(data, iteration, eval_mode=False)
  File "/usr/local/lib/python3.10/dist-packages/ray/util/tracing/tracing_helper.py", line 467, in _resume_span
    return method(self, *_args, **_kwargs)
  File "/root/ChatLearn/examples/megatron/models/old_policy_inference.py", line 362, in _forward_step
    tokens, all_log_probs = self.generate(
  File "/root/ChatLearn/examples/megatron/models/old_policy_inference.py", line 290, in generate
    res = generate_tokens_probs_and_return_on_first_stage(
  File "<string>", line 205, in generate_tokens_probs_and_return_on_first_stage 
  File "/root/Megatron-LM/megatron/inference/text_generation/communication.py", line 95, in broadcast_from_last_to_first_pipeline_stage
    _is_cuda_contiguous(tensor)
  File "/root/Megatron-LM/megatron/inference/text_generation/communication.py", line 55, in _is_cuda_contiguous
    assert tensor.is_contiguous()
AssertionError
```

This is because Megatron-LM does not set `output_log_probs` to a contiguous tensor when pipeline parallelism is enabled. You can refer to [NVIDIA/Megatron-LM#570](https://github.com/NVIDIA/Megatron-LM/pull/570) for a quick fix.


## Applying for custom_port
In the DLC environment, the current RLHF training has already allocated 50 ports to meet all usage scenarios. It is recommended to set the advanced configuration as follows:
```
customPortList=30000-30050
```
## Task failure but DLC status shows success
1. Redirect the log to a file
```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${output_dir}/log_${RANK}.txt
```
In this situation, the exit code is always 0, and the DLC job will show as successful. It is necessary to change it to the following:
```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${output_dir}/log_${RANK}.txt ; exit ${PIPESTATUS[0]}
```
2. There are some additional operations after the training command, causing the error code to be different from the training command's error code. It is recommended to add `set -e` at the beginning of the command, so that it exits at the first encountered error command.
## Adjusting lr error in continued training
Megatron checks if the lr has changed during load_checkpoint. It is necessary to set the Megatron model parameter `override_opt_param_scheduler` to True to bypass the check.
## How to specify the frequency of model saving during training
In rlhf.yaml, configure `save_episode_interval`.
