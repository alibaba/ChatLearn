from typing import List, Tuple, Union
from vllm.model_executor.models.registry import _ModelRegistry, _LazyRegisteredModel, _ModelInfo

def inspect_model_cls(
    self,
    architectures: Union[str, List[str]],
) -> Tuple[_ModelInfo, str]:

    ###### additional code ######
    self.models["Qwen3ForCausalLM"] = _LazyRegisteredModel(
        # module_name=f"vllm.model_executor.models.qwen3",
        module_name="chatlearn.models.vllm.hooks.vllm_0_6_6.qwen3",
        class_name="Qwen3ForCausalLM",
    )
    self.models["Qwen3MoeForCausalLM"] = _LazyRegisteredModel(
        # module_name=f"vllm.model_executor.models.qwen3_moe",
        module_name="chatlearn.models.vllm.hooks.vllm_0_6_6.qwen3_moe",
        class_name="Qwen3MoeForCausalLM",
    )
    ###### additional code ######
    
    architectures = self._normalize_archs(architectures)

    for arch in architectures:
        model_info = self._try_inspect_model_cls(arch)
        if model_info is not None:
            return (model_info, arch)

    return self._raise_for_unsupported(architectures)

_ModelRegistry.inspect_model_cls = inspect_model_cls

