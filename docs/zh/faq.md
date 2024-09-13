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

## 申请 custom_port

在 DLC 环境中，当前RLHF训练申请50个port已经满足所有使用场景。建议设置高级配置如下：

```
customPortList=30000-30050
```

## 任务失败但是DLC状态显示成功

1. 将log重定向到文件

```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt
```

这种情况退出码总是0，DLC作业会显示成功。需要加上改成以下方式

```
python train_rlhf.py -c configs/llama2/rlhf.yaml 2>&1 | tee -a ${LOG_DIR}/log_${RANK}.txt ; exit ${PIPESTATUS[0]}
```

2. 在训练命令之后有一些额外的操作，导致错误码非训练命令的错误码。建议在 command 开头加上 `set -e`, 这样就可以在第一个遇到错误的命令退出。

## 续跑调整 lr 报错

Megatron在load_checkpoint的时候会检查lr是否变化，需要设置 Megatron 模型参数 `override_opt_param_scheduler` 为True 来绕开检查。

## 如何指定训练时模型保存的频率

rlhf.yaml 里配置  `save_episode_interval`。
