# FAQ

## Converting Models Between Megatron and Hugging Face Formats

[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) provides comprehensive support for converting model formats between Mcore and Hugging Face (HF). Refer to the [documentation](https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/distributed_checkpoints_convertor) for detailed instructions on model conversion. For example, to convert the DeepSeek 671B model, run the following command:

```bash
bash scripts/deepseek_v3/run_32xH20.sh \
A37B \
/mnt/deepseek-ckpts/DeepSeek-V3-bf16 \
/mnt/deepseek-ckpts/DeepSeek-V3-to-mcore \
false \
true \
bf16
```

## How to Accelerate FSDP Training Speed

1. Set `models.policy_trainer.packing=True`, and set `models.policy_trainer.max_token_in_packing` to the maximum total number of tokens that fully utilizes GPU memory.
2. For Qwen3-MoE models, set `models.policy_trainer.groupgemm=True` to enable the GroupGEMM patch, which improves training speed of MoE layers.

# Common Errors

## `ray.exceptions.RayChannelTimeoutError` During Rollout Inference

```bash
ray.exceptions.RayChannelTimeoutError: System error: If the execution is expected to take a long time, increase RAY_CGRAPH_get_timeout which is currently 10 seconds. Otherwise, this may indicate that the execution is hanging.
```

When this error occurs during vLLM rollout inference, check whether `models.policy.tensor_model_parallel_size` is not equal to 1. If `tensor_model_parallel_size` is greater than 1, set `models.policy.enforce_eager=True`.

## Why does Ray report an OOM error when loading weights with transformers during FSDP model initialization?

Enable `models.policy_trainer.meta_init=True`. Note that this may increase initialization time.

## Why does `torch.OutOfMemoryError: CUDA out of memory` occur during FSDP training?

1. If `models.policy_trainer.packing=True`, try reducing `models.policy_trainer.max_token_in_packing`.
2. If `models.policy_trainer.packing=False`, try reducing `runtime_args.train_micro_batch_size`.
3. If `runtime_args.train_micro_batch_size=1`, or if OOM still occurs even when `models.policy_trainer.max_token_in_packing` is smaller than the generation length, consider increasing `models.policy_trainer.ulysses_sequence_parallel_size`. It is recommended to set it to a power of 2 and not exceed the number of GPUs per node.

## Why Does RL Training of the Megatron-Core MoE Model Suffer from Corruption or Poor Performance?

Megatron-Core is primarily designed for pretraining and SFT of LLMs, and does not fully account for certain numerical precision concerns that are critical during RL training. As a result, these oversights may lead to instability or degraded performance in RL scenarios. Below are key issues we have identified in MoE model training—consider modifying the source code if you encounter convergence problems.

1. The `unpermute()`in `moe_utils.py` may perform a scatter-add in `bfloat16`. This low-precision accumulation is numerically unstable and can cause inconsistent outputs between two forward passes on the same data batch. To mitigate this:
+ Ensure that `moe_permute_fusion` is `False` in your log file (indicating the fused kernel is disabled).
+ Upgrade the computation precision of this operation to `fp32` or even `fp64` to improve numerical stability (More GPU Memory required). 

2. The parameters of MoE routers can significantly affect the logit output, especially in off-policy RL settings. Small changes in routing decisions can amplify distributional shifts. Consider the following adjustments:
   + Reduce the scaling factor of the router load balancing loss, or disable it entirely.
   + If router bias is enabled (e.g., in models like `DeepSeek-V3` or `Moonlight`), reduce the `moe_router_bias_update_rate` to prevent overly aggressive updates that destabilize the policy during RL fine-tuning.
