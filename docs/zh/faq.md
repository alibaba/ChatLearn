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

## 为什么 Megatron-Core 的 MoE 模型在强化学习（RL）训练中会出现性能下降或结果异常？

Megatron-Core 主要针对大语言模型（LLM）的预训练和监督微调（SFT）进行设计，未充分考虑强化学习（RL）训练中对数值精度的特殊要求。这些缺失可能导致 RL 训练过程中出现不稳定或性能退化等问题。以下是我们在 MoE 模型训练中发现的关键问题。若你遇到收敛困难，建议对源码进行相应修改。

MoE 路由器的参数可能显著影响模型的 logits 输出，尤其是在离线策略（off-policy）训练场景下。路由行为的微小变化可能引发较大的分布偏移。建议进行以下调整：
   + 降低路由器负载均衡损失（router load balance loss）的权重系数，或直接关闭该损失项，以防其干扰策略稳定性。
   + 如果模型中启用了router bias（如 `DeepSeek-V3` 或 `Moonlight`），应降低 `moe_router_bias_update_rate`，避免在 RL 微调阶段因路由更新而导致的崩溃。

具体来讲，在chatlearn/utils/megatron_utils.py中调整降低路由器负载均衡损失（router load balance loss）的权重系数，或直接关闭该损失项，以防其干扰策略稳定性。MoE模型中控制负载均衡相关的参数 `moe_router_load_balancing_type` 和 `moe_aux_loss_coeff` 可能对模型训练稳定性有较大影响。
```bash
#cfg.models.policy_trainer.megatron_model_cfg.moe_router_load_balancing_type = "seq_aux_loss"
#cfg.models.policy_trainer.megatron_model_cfg.moe_aux_loss_coeff = 0.001
cfg.models.policy_trainer.megatron_model_cfg.moe_router_load_balancing_type = "none"
cfg.models.policy_trainer.megatron_model_cfg.moe_aux_loss_coeff = 0
```

将chatlearn/utils/megatron_utils.py中的`moe_router_enable_expert_bias`关闭以及`moe_router_bias_update_rate`从1e-3修改为0，避免在RL微调阶段因路由更新而导致的崩溃。
```bash 
#cfg.models.policy_trainer.megatron_model_cfg.moe_router_enable_expert_bias = True
cfg.models.policy_trainer.megatron_model_cfg.moe_router_enable_expert_bias = False
#cfg.models.policy_trainer.megatron_model_cfg.moe_router_bias_update_rate =1e-3
cfg.models.policy_trainer.megatron_model_cfg.moe_router_bias_update_rate = 0.0
```

