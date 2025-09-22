"""utils function megatron policy trainer"""
from .policy_model import GPTPolicyModel
try:
    from .policy_model_vl import Qwen2_5VLPolicyModel
except ImportError:
    import warnings
    warnings.warn("VL needs megatron_patch. Please set env var MEGATRON_PATCH_PATH to include megatron_patch")
from .train_helper import forward_step, training_log
