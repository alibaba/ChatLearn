# 常见问题

## ECC Error

ECC Error 为机器故障，建议结合 [续跑和容错](tutorial/continue_train.md) 来自动拉黑故障机器并重启作业。

## 如何构建多个reward模型的自定义训练flow

当前提供的例子里为单个reward模型的训练范例，如果需要自定义多个reward模型的训练flow，可以参考 [自定义推理和训练流程](tutorial/custom_model_flow.md)。

## RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding

```
RuntimeError: Error(s) in loading state_dict for VocabParallelEmbedding:
   size mismatch for weight: copying a param with shape torch.Size([xxx, xxx]) from checkpoint, the shape in current model is torch.Size([[xxx, xxx]]).
```
这个一般是因为改变了TP引起的，需要调整 `make_vocab_size_divisible_by` 这个参数来对齐padding后的embedding参数shape。

## yaml 配置

参考 [配置文件](config_yaml.md)。

## 如何开启 `Efficient memory sharing` 功能来减少显存开销

参考文档 [Efficient memory sharing](tutorial/ems.md)。

## Megatron 模型转换并行策略

```bash
cd $CHATLEARN
model_type=GPT # for reward model, model_type set to REWARD
load_dir=xxx
save_dir=xxx
target_tp=xxx
target_pp=xxx
python chatlearn/tools/megatron_checkpoint_utils.py --model-type ${model_type} --load-dir ${load_dir} --save-dir ${save_dir} \
    --target-tensor-parallel-size ${target_tp} --target-pipeline-parallel-size ${target_pp}
```

注意这个脚本只在官方Megatron-LM脚本上验证过。

## 转换 Checkpoint 失败
使用 Megatron-LM 版本 core_r0.8.0 作为后端来转换 Checkpoint 可能会导致以下错误：

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

此问题是在 Megatron-LM 的 Checkpoint 转换代码中默认进程组未初始化。它是在 Megatron-LM core_r0.8.0 版本中引入的。我们目前有如下两种可能的方案来解决这个问题：

1. 您可以注释掉有问题的那一行，因为这仅影响调试级别的日志输出。
2. 考虑使用 Megatron-LM 版本 core_r0.9.0 作为后端，因为此版本已经修复了这个 bug。然而，该版本的正确性和性能尚未在 ChatLearn 中得到验证。我们计划在未来升级 Megatron-LM 版本到 core_r0.9.0。

## Alignment 训练开启流水线并行时存在非 contiguous 的张量

若使用 Megatron-LM 作为后端进行 alignment 训练并开启流水线并行，可能会导致如下错误：

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

这是因为 Megatron-LM 未把 `output_log_probs` 设置为 contiguous 的张量。您可以参考 [NVIDIA/Megatron-LM#570](https://github.com/NVIDIA/Megatron-LM/pull/570) 进行修复。

## 申请 custom_port

在 DLC 环境中，当前RLHF训练申请50个port已经满足所有使用场景。建议设置高级配置如下：

```
customPortList=30000-30050
```

## 任务失败但是DLC状态显示成功

1. 将log重定向到文件

```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${output_dir}/log_${RANK}.txt
```

这种情况退出码总是0，DLC作业会显示成功。需要加上改成以下方式

```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${output_dir}/log_${RANK}.txt ; exit ${PIPESTATUS[0]}
```

2. 在训练命令之后有一些额外的操作，导致错误码非训练命令的错误码。建议在 command 开头加上 `set -e`, 这样就可以在第一个遇到错误的命令退出。

## 续跑调整 lr 报错

Megatron在load_checkpoint的时候会检查lr是否变化，需要设置 Megatron 模型参数 `override_opt_param_scheduler` 为True 来绕开检查。

## 如何指定训练时模型保存的频率

rlhf.yaml 里配置  `save_episode_interval`。
