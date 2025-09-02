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