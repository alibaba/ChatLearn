# End-to-End GRPO Training Tutorial with FSDP

This document provides instructions for end-to-end training using the ChatLearn, pytorch FSDP and vLLM framework, and the qwen3 model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task?spm=a2c4g.11186623.help-menu-30347.d_3_3_5_5.2dfb1925l3QjwG). You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`.

2. Code Preparation

```bash
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## Data Preparation
We take [MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval) as exmaple.
```bash
# download dataset
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# preprocess dataset
python chatlearn/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
```

## Training
You can run the following command to start training:

### Qwen3-8B
Run this command on server with 8 GPUs
```bash
# download model weight
modelscope download --model Qwen/Qwen3-8B --local_dir pretrained_models/Qwen3-8B
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the configuration with: 
```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
Change the configuration to:
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```

## Model Conversion
Saving FSDP models is time-consuming. Chatlearn provides an offline model conversion feature, which converts FSDP-sharded checkpoints back to HuggingFace format. The script is as follows:
```bash
export CHATLEARN=$(pwd)
python chatlearn/offline_ckpt_converter.py \
    --hf_dir ${CHATLEARN}/Qwen3-8B/ \
    --ckpt_dir ${CHATLEARN}/output/qwen3-grpo-8b/save_model/policy_trainer \
    --save_dir ${CHATLEARN}/output/qwen3-grpo-8b/save_model/huggingface/ \
    --iter 200 \
    --groupgemm 0
```
If you are training an MoE model with groupgemm, please make sure to set:
```bash
   --groupgemm 1
```
This script will convert the final FSDP sharded model after training back into a HuggingFace model and save it in the path "${CHATLEARN}/output/qwen3-grpo-8b/save_model/huggingface/".

## FAQ
### How to Speed Up PolicyTrainer Training?
1. Set models.policy_trainer.packing=True and configure models.policy_trainer.max_token_in_packing to the maximum token count that fits GPU memory.

2. For the Qwen3-MoE model, enable models.policy_trainer.groupgemm=True to activate the GroupGEMM patch, improving MoE layer training speed.

### Why Does FSDP Initialization Cause Ray OOM Errors When Load Weights in Transformers?
Enable models.policy_trainer.meta_init=True to mitigate this issue. This may cause extra time cost for initialization.

### Why Does This Error Occur During Inference?
```bash
ray.exceptions.RayChannelTimeoutError: System error: If the execution is expected to take a long time, increase RAY_CGRAPH_get_timeout which is currently 10 seconds. Otherwise, this may indicate that the execution is hanging.
```
Check the model input parameter: If models.policy.tensor_model_parallel_size is not 1, set models.policy.enforce_eager=True.

### Why Does torch.OutOfMemoryError: CUDA Out of Memory Occur During Training?
1. If models.policy_trainer.packing=True, try reducing models.policy_trainer.max_token_in_packing.

2. If models.policy_trainer.packing=False, decrease runtime_args.train_micro_batch_size.

3. If OOM persists even with runtime_args.train_micro_batch_size=1 or when models.policy_trainer.max_token_in_packing is smaller than the generation length, increase models.policy_trainer.ulysses_sequence_parallel_size (recommended: a power of 2, not exceeding the number of GPUs per node).


### Why Does CUDA OOM Still Occur After These Adjustments?
Consider scaling up the number of GPUs—FSDP memory consumption scales roughly linearly with the total GPU count.

### Why Does vLLM Initialization Cause CUDA OOM?
Increase models.policy.gpu_memory_utilization (recommended: no higher than 0.95).
