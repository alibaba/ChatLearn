from old_policy_megatron_inference import PolicyMegatronInference
from tqdm import tqdm
from train_rlhf import get_prompts_from_sft

import rlhf
from rlhf import EvalEngine
from utils.utils import write_jsonl

rlhf.init()

policy = PolicyMegatronInference("policy")
engine = EvalEngine(policy)
engine.register_func("policy", "forward_step")

args = rlhf.get_args()
k = {"math_coef": 0}
train_prompts = get_prompts_from_sft(args.rlhf_args._args_dict["eval_data_path"], policy.model_args, num_limit=10, )

policy_checkpoint = policy.model_args["load"]
load_iteration = policy.model_args.get("load_iteration", 0)
exp_name = policy.model_args["exp_name"]
eval_dir = args.rlhf_args._args_dict["eval_output_dir"]

engine.set_dataset(train_prompts)
results = engine.eval()
output = []

for res in tqdm(results, total=len(results)):
    print(res["str_outputs"])
    str_prompts = res["str_prompts"]
    str_outputs = res["str_outputs"]
    for str_prompt, str_output in zip(str_prompts, str_outputs):
        j = {"query": str_prompt, "responses": [str_output]}
        output.append(j)

policy_inference_fp = f"{eval_dir}/{load_iteration}/{exp_name}/inference_json.json"
print(policy_inference_fp)
print(f"inference finished: got jsons number: {len(output)}")
write_jsonl(output, policy_inference_fp)

engine.logging_summary()
