# Debugging Parameter Synchronization

The ParameterSync module ensures weight consistency between the training (trainer) and serving (inference) components. This guide helps debug synchronization issues that may arise when using different distributed strategies, such as:

Trainer Side: Megatron-LM (expert parallel + tensor parallel + pipeline parallel)
Inference Side: vLLM (tensor parallel)

## Step-by-Step Debugging Guide

### 1. Set Up Environment Variable

Specify a path to dump parameter snapshots before/after synchronization by setting the DEBUG_SYNC_PARAMETERS_PATH environment variable, and huggingface format checkpoint used in vLLM:

``` bash
export DEBUG_SYNC_PARAMETERS_PATH=/path/to/dump_directory
export vllm_load_format=auto
export policy_load=/workspace/hf_ckp/QWen2-Max
```

### 2. Launch the ChatLearn Job

Start your training job (e.g., DPO fine-tuning for Llama 2) using [tutorial_llama2.md](/docs/en/tutorial/tuotiral_llama2.md):
```bash
bash scripts/train_online_dpo_llama.sh
```

This will generate two directories:
before_sync_parameter: Parameters before synchronization.
after_sync_parameter: Parameters after synchronization.
Each directory contains subfolders for every TP rank.

### 3. Check Dumped Files
Verify the dumped parameter files:
``` bash
tree /path/to/dump_directory
```

Example output:
``` text
     /workspace/debug_sync_params
     ├── before_sync_parameter
     │   ├── 0   # Parameters from TP rank 0
     │   ├── 1   # Parameters from TP rank 1
     │   └── ...
     └── after_sync_parameter
         ├── 0
         ├── 1
         └── ...
```

### 4. Run the Parameter Check Script
Use the check_sync_parameter.py tool to compare parameters before/after synchronization:
``` bash
python chatlearn/tools/check_sync_parameter.py --root_dir /path/to/dump_directory | tee check.log
```

This script will compare parameter shapes and values

Generate a log file (check.log) with detailed results.

### 5. Interpret the Log File

Successful Sync:
```text
PASS|1|model.layers.5.self_attn.qkv_proj.weight
```

MisMatch Detected with Mean Values:
```text
DIFF|1|model.layers.3.mlp.shared_expert.gate_up_proj.weight|torch.Size([2816, 2048])|tensor(0.5247)|torch.Size([2816, 2048])|tensor(8.231)
```