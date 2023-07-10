from old_policy_megatron_inference import PolicyMegatronInference
from reward_inference_module_hongyi import RewardModelMegatronInference

import rlhf
from rlhf import EvalEngine
from train_rlhf import get_prompts
from utils.utils import write_jsonl

rlhf.init()

policy = PolicyMegatronInference("policy")
reward = RewardModelMegatronInference("reward")
engine = EvalEngine([policy, reward])

# TODO: read from files
# train_prompts = ["某医院将为一位退休的资深医生举行感恩晚宴,请为晚宴设计一段开场白。", "某医院将为一位退休的资深医生举行感恩晚宴,请为晚宴设计一段开场白。"] * 32

all_prompts = get_prompts(rlhf.get_args().rlhf_args.data_path, num_limit=None)
wangwei_600_prompts = all_prompts[200:]

wangwei_len = len(wangwei_600_prompts)

engine.set_dataset(wangwei_600_prompts)
engine.register_func("policy", "forward_step")
engine.register_func("reward", "eval_forward")
list_json_res = engine.eval()

policy_checkpoint = policy.model_args["load"]
load_iteration = policy.model_args["load_iteration"]
exp_name = policy.model_args["exp_name"]

wangwei_res = list_json_res
policy_wangwei_res_fp = f"{policy_checkpoint}/eval_dir/{load_iteration}/{exp_name}/eval_json_all_res.json"
print(f"eval inference finished: got jsons number: {wangwei_len}")
write_jsonl(wangwei_res, policy_wangwei_res_fp)
