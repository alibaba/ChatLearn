def apply_monkey_patch(model_config, op_list):
    print(f"apply {op_list} patches for {model_config.architectures}")
    if model_config.architectures[0] == "Qwen2ForCausalLM":
        if "sequence_parallel" in op_list:
            from chatlearn.models.patches.transformers.qwen2_patch import register_sp_attention_forward
            register_sp_attention_forward()
    if model_config.architectures[0] == "Qwen3ForCausalLM":
        if "sequence_parallel" in op_list:
            from chatlearn.models.patches.transformers.qwen3_patch import register_sp_attention_forward
            register_sp_attention_forward()
    else:
        raise ValueError(f"Unsupported model architecture: {model_config.architectures}")
