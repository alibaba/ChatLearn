"""utils function megatron policy trainer"""
from .policy_model import GPTPolicyModel
try:
    from .policy_model_vl import Qwen2_5VLPolicyModel
except ImportError:
    import warnings
    warnings.warn("megatron_patch is not installed.")
from .train_helper import forward_step, training_log
