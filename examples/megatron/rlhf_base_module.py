from megatron import get_args
from megatron.initialize import initialize_megatron

from rlhf import RLHFMegatronModule
from utils.utils import read_latest_ppo_iter


class MegatronInferenceModule(RLHFMegatronModule):

    def set_args_for_continue_train(self, name):
        policy_train_global_batch_iter = read_latest_ppo_iter(name)
        self.args.iteration_for_log = policy_train_global_batch_iter * self.args.continue_train_global_batch_size // (
                self.args.continue_inference_instances * self.args.continue_inference_batch_size)
        self.args.load = f"{get_args().save}/{name}/{get_args().exp_name}"
        self.args.load_iteration = policy_train_global_batch_iter

        print(f"continue train iteration_for_log: {self.args.iteration_for_log}")
        print(f"continue train self.args.load: {self.args.load} load_iteration {self.args.load_iteration}")
        self.args.no_load_optim = False  # latest
        self.args.no_load_rng = False  # latest
        self.args.no_load_args = False  # latest
        self.args.no_load_scheduler = False  # latest


class MegatronTrainingModule(RLHFMegatronModule):

    def init(self):
        args_dict = self.model_args
        initialize_megatron(args_defaults={}, ignore_unknown_args=True,
                            args_dict=args_dict)
        self.stats = {}
        self.buffer = {}
