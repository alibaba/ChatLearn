# 常见问题

## Megatron和HF模型相互转换

[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)对Mcore与HF之间模型格式的转换做了完善的支持，可参考[文档](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/distributed_checkpoints_convertor)进行模型转换。如deepseek 671B可运行如下命令进行转换。

```bash
bash scripts/deepseek_v3/run_32xH20.sh \
A37B \
/mnt/deepseek-ckpts/DeepSeek-V3-bf16 \
/mnt/deepseek-ckpts/DeepSeek-V3-to-mcore \
false \
true \
bf16
```

## 如何加快FSDP的训练速度

1. 设置models.policy_trainer.packing=True，并设置models.policy_trainer.max_token_in_packing=可以打满显存的总token数。
2. 如果是qwen3-moe模型，可以设置models.policy_trainer.groupgemm=True，打开groupgemm patch，提升moe层训练速度。

# 常见报错

## Rollout推理过程中出现ray.exceptions.RayChannelTimeoutError
```bash
ray.exceptions.RayChannelTimeoutError: System error: If the execution is expected to take a long time, increase RAY_CGRAPH_get_timeout which is currently 10 seconds. Otherwise, this may indicate that the execution is hanging.
```
在vLLM Rollout推理过程中出现如下报错，可以models.policy.tensor_model_parallel_size是否不为1，若tensor_model_parallel_size不为1，建议设置models.policy.enforce_eager=True。

## 为什么FSDP模型初始化时，在transfomers读权重文件的时候ray有oom报错？

可以打开models.policy_trainer.meta_init=True。这可能会带来额外的初始化时间

## 为什么在FSDP训练时会出现，torch.OutOfMemoryError: CUDA out of memory？
1. 如果models.policy_trainer.packing=True，尝试降低models.policy_trainer.max_token_in_packing。
2. 如果models.policy_trainer.packing=False，尝试降低runtime_args.train_micro_batch_size。
3. 如果runtime_args.train_micro_batch_size=1，或models.policy_trainer.max_token_in_packing小于生成长度依然oom，则建议提高models.policy_trainer.ulysses_sequence_parallel_size，推荐设置为2的倍数，且不超过单机卡数。