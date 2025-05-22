from dataclasses import dataclass, field
# from chatlearn.configs.common import (
from configs.common import (
    RuntimeEnvConfig,
    PolicyConfig,
    RuntimeConfig,
    RewardConfig,
    RefPolicyConfig,
    PolicyTrainerConfig,
)
# from chatlearn.algorithm.base_algo import BaseAlgorithm
from algorithm.base_algo import BaseAlgorithm


@dataclass
class GrpoConfig:
    num_episodes: int = field(
        default=200,
        metadata={"help": "Number of episodes to train."}
    )
    sample_per_episode: int = field(
        default=1024,
        metadata={"help": "Number of samples per episode."}
    )
    runtime_env: RuntimeEnvConfig = field(
        default_factory=RuntimeEnvConfig,
        metadata={"help": "Runtime environment config."}
    )
    runtime: RuntimeConfig = field(
        default_factory=RuntimeConfig,
        metadata={"help": "Runtime config."}
    )
    policy: PolicyConfig = field(
        default_factory=PolicyConfig,
        metadata={"help": "Policy config."}
    )
    reward: RewardConfig = field(
        default_factory=RewardConfig,
        metadata={"help": "Reward config."}
    )
    ref_policy: RefPolicyConfig = field(
        default_factory=RefPolicyConfig,
        metadata={"help": "Reference policy config."}
    )
    policy_trainer: PolicyTrainerConfig = field(
        default_factory=PolicyTrainerConfig,
        metadata={"help": "Policy trainer config."}
    )


class GrpoAlgorithm(BaseAlgorithm):

    def __init__(self, cfg: GrpoConfig) -> None:
        self.cfg = cfg


    def run(self) -> None:
        print(self.cfg)
        # chatlearn.init(self.cfg)
        # policy_trainer = PolicyTrainer("policy_trainer")
        # ref_policy = PolicyTrainer("ref_policy")
        # policy = VLLMPolicyInference("policy")
        # reward = RuleReward("reward")
        # engine = GRPOEngine(policy, reward, ref_policy, policy_trainer)

        # # get train and evaluation data
        # train_data_path_list = [item.strip() for item in args.runtime_args.data_path.split(",")]
        # train_data = read_data_path_list(train_data_path_list)

        # eval_data_path_list = [item.strip() for item in args.runtime_args._args_dict["eval_data_path"].split(',')]
        # eval_data = read_data_path_list(eval_data_path_list)

        # # put data in engine._all_datasets
        # engine.set_dataset(train_data)
        # engine.evaluator.set_dataset(eval_data)
        # engine.set_relay_sample_manager(compute_grpo_adv)
        # engine.learn()