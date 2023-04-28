"""examples"""
import torch
from gpt_megatron import GPTMegatron
from gpt_allspark import GPTAllSpark
import rlhf
from rlhf import RLHFEngine
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class RLHFDataset(Dataset):
    def __init__(self, train_path, tokenizer, max_length):
        self.post_list = []
        dataset = load_dataset(train_path, split='train')
        num_samples = 2000
        for i, sample in enumerate(dataset):
            if i == num_samples:
                break
            self.post_list.append(sample["prompt"] + sample["label"])
            
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.ones([self.max_length], dtype=torch.int64)
        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
        }


class PolicyModel(GPTAllSpark):
    
    def forward_step(self, data):
        return {'input_ids': data["input_ids"],
                'response_ids': data["input_ids"],
                'response_mask': data["attention_mask"],
                'old_log_probs': data["input_ids"]}


class ReferenceModel(GPTAllSpark):

    def forward_step(self, data):
        return {'ref_log_probs': data["input_ids"]}


class ValueModel(GPTAllSpark):

    def forward_step(self, data):
        response = data["response_ids"]
        return {"old_values": response}


class RewardModel(GPTAllSpark):
    
    def forward_step(self, *data):
        policy_input = data[0]
        ref_input = data[1]
        value_input = data[2]
        bs = policy_input['input_ids'].size(0)
        rewards = torch.ones([bs]).cuda()
        advantage = torch.ones([bs]).cuda()
        total_rewards = torch.ones([bs]).cuda()
        return {"rewards": rewards, "advantage": advantage, "total_rewards": total_rewards}


class PPOPolicyModel(GPTMegatron):
    
    def train_step(self, data):
        return 1


class PPOValueModel(GPTMegatron):
    
    def train_step(self, data):
        return 1


if __name__ == "__main__":
    rlhf.init()
    args = rlhf.get_args()

    models = {}

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = RLHFDataset("CarperAI/openai_summarize_tldr", tokenizer, 2048)
    policy_model = PolicyModel("policy")
    value_model = ValueModel("value")
    reference_model = ReferenceModel("reference")
    reward_model = RewardModel("reward")
    ppo_policy_model = PPOPolicyModel("ppo_policy")
    ppo_value_model = PPOValueModel("ppo_value")

    engine = RLHFEngine(policy_model, reference_model, reward_model, value_model, ppo_policy_model, ppo_value_model)
    engine.set_dataset(dataset)
    engine.learn()
