"""GPT"""
import datetime
import os
import time
from functools import partial

from megatron.core.enums import ModelType
from megatron.training import print_datetime
from megatron.training import train_step as megatron_train_step
from rlhf_megatron_training import setup_model_and_optimizer

from .constants_ppo import select_actions_from_right_padded, get_ltor_masks_and_position_ids, pad_to_max_len

_TRAIN_START_TIME = time.time()

import numpy as np
import torch

from models.policy_model import PolicyModel
from utils.utils import training_log
from megatron import get_args, get_num_microbatches
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu
from megatron import print_rank_0
from megatron.initialize import set_jit_fusion_options

from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import calc_params_l2_norm
from rlhf.utils import to_device
from rlhf import RLHFMegatronModule


class AdaptiveKLController:
    """Adaptive KL Controller as described in Ziegler et al. "Fine-Tuning Language Models from Human Preferences"
    Reference: Section 2.2 https://arxiv.org/pdf/1909.08593.pdf#page=2
    Source: https://github.com/openai/lm-human-preferences/blob/master/lm_human_preferences/train_policy.py
    """

    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int):
        """Returns adaptively updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        proportional_error = np.clip(current / self.target - 1, -0.2, 0.2)  # ϵₜ
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult  # βₜ₊₁


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current: float, n_steps: int):
        """Returns updated KL coefficient, βₜ₊₁.
        Arguments:
            current: The current KL value between the newest policy and the initial policy.
        """
        pass


def cyclic_iter(iter):
    while True:
        for x in iter:
            yield x


class MegatronPolicy(RLHFMegatronModule):
    """gpt model wrapper"""

    def setup(self):
        self.buffer = {}
        self.stats = {}
        self.report_memory_flag = True

        # param_ranks = [ranks[0] for ranks in mpu.get_all_data_parallel_group_ranks()]
        # self.set_param_ranks(param_ranks)

        self.args = get_args()

        self.model_type = ModelType.encoder_or_decoder
        self.tokenizer = get_tokenizer()

        # always a adaptive one but may not use this value but kept at init_kl_coeff
        self.kl_ctl = AdaptiveKLController(
            self.args.init_kl_coef, self.args.target, self.args.horizon
        )
        # else:
        #     self.kl_ctl = FixedKLController(self.args.init_kl_coef)

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        # Adjust the startup time so it reflects the largest value.
        # This will be closer to what scheduler will see (outside of
        # image ... launches.
        global _TRAIN_START_TIME
        start_time_tensor = torch.cuda.DoubleTensor([_TRAIN_START_TIME])
        torch.distributed.all_reduce(start_time_tensor,
                                     op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
            time.time() - _TRAIN_START_TIME))
        print_datetime('after megatron is initialized')

        timers = get_timers()

        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup').start()

        print(f"policy trainer loading : {self.args.load}")
        get_args().save = f"{get_args().save}/policy/{get_args().exp_name}"
        if self.args.continue_train:
            self.args.load = get_args().save
            self.args.load_iteration = -1  # latest

            self.args.no_load_optim = False  # latest
            self.args.no_load_rng = False  # latest
            self.args.no_load_args = False  # latest
            self.args.no_load_scheduler = False  # latest

            print(
                f"policy trainer continue train args load: {self.args.load} self.args.load_iteration {self.args.load_iteration}")

        self.model, self.optimizer, self.opt_param_scheduler = setup_model_and_optimizer(self.model_provider,
                                                                                         self.model_type)

        timers('model-and-optimizer-setup').stop()
        timers.log(['model-and-optimizer-setup'])

        if int(os.environ.get("WORLD_SIZE", 1)) > 1:
            torch.distributed.barrier(device_ids=[int(os.environ.get("LOCAL_RANK", 0))])

        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.run_name = f"policy_trainer run-{timestamp}"

        print(f"End setup PPO megatron", flush=True)

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        model = PolicyModel(
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
            stats=self.stats
        )
        return model

    def get_batch(self, batch_data):
        """Generate a batch"""
        args = self.args

        # Items and their type.
        '''
                "all_token_ids_right_padded": torch.tensor([[p,p,5,6,7], [p,p,p,8,9]], dtype=torch.long, device=device),
                "action_start_indices": torch.tensor([[10,100,p,p,p], [11,p,p,p,p]], dtype=torch.long, device=device),
                "action_logprobs": torch.randn([bs, 5], dtype=torch.float32, device=device),
                "action_values": torch.randn([bs, 5], dtype=torch.float32, device=device),
                "action_rewards": torch.randn([bs, 5], dtype=torch.float32, device=device),
                "loss_mask"
        '''
        int64_keys = ['all_token_ids_right_padded', 'action_start_indices']
        float32_keys = ["action_logprobs", "action_values", 'action_rewards']

        data_b = next(batch_data)

        tokenizer = get_tokenizer()

        # TODO tianhang move to sayang's framework later. add pad to max length config
        all_token_ids_right_padded = pad_to_max_len(data_b["all_token_ids_right_padded"], args.seq_length,
                                                    pad_value=get_tokenizer().eod_id)
        all_token_loss_mask = pad_to_max_len(data_b["loss_mask"], args.seq_length, pad_value=0)

        all_token_attention_mask, all_token_position_ids = get_ltor_masks_and_position_ids(
            all_token_ids_right_padded)
        # print(f"all_token_position_ids: {all_token_position_ids}")
        response_length = data_b["action_rewards"].shape[1]

        inputs = {
            "all_token_position_ids": all_token_position_ids,
            "all_token_ids_right_padded": all_token_ids_right_padded,
            # this attention mask is not TRANSFOEMRER attention msak. this actually applies on attention result [b, np, s, s]
            "all_token_attention_mask": all_token_attention_mask.bool(),
            "all_token_loss_mask": all_token_loss_mask.bool(),

            "action_starts": data_b['action_start_indices'],

            "action_logprobs": data_b["action_logprobs"].float(),  # response size
            "action_values": data_b["action_values"].float(),
            "action_rewards": data_b["action_rewards"].float(),

        }

        assert all_token_ids_right_padded.size(0) != 1, f"cannot be 1 will be squeezed. {all_token_ids_right_padded}"

        for k, v in inputs.items():
            inputs[k] = to_device("cuda", v)
            # print(f"{k} size {inputs[k].size()}")

        #         print(f"{k}: size: {v.size()}")
        return inputs

    def aggregate_loss_func(self, inputs, losses):  # [b, s]
        # losses = losses.float()
        # b = losses.size(0)
        # loss = torch.sum(losses.view(-1)) / b

        losses = losses.float()  # [b, response_size]

        old_rewards = inputs['action_rewards']  # [b, responses size]
        response_length = old_rewards.shape[1]
        # Note the tkoken logits to get loss is only the actions. query doesn't have loss.
        action_loss_mask = select_actions_from_right_padded(ts=inputs["all_token_loss_mask"],
                                                            action_starts=inputs["action_starts"] - 1,
                                                            # because align iwth logits index
                                                            response_size=response_length,
                                                            pad_value=0,
                                                            dim=-1).contiguous()

        action_loss_mask = action_loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * action_loss_mask) / action_loss_mask.sum()

        # Reduce loss for logging.
        averaged_loss = average_losses_across_data_parallel_group([loss])
        # Reduce loss for logging.
        self.stats["policy_loss"] = averaged_loss[0]
        return loss, {'policy lm loss': averaged_loss[0]}

    def _forward_step(self, batch_data, model):
        """Forward step."""
        # args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator').start()
        inputs = self.get_batch(
            batch_data)
        timers('batch-generator').stop()

        losses = model.forward(all_token_ids=inputs["all_token_ids_right_padded"],
                               all_position_ids=inputs["all_token_position_ids"],
                               all_token_attention_mask=inputs["all_token_attention_mask"],
                               training_inputs=inputs)

        return losses, partial(self.aggregate_loss_func,
                               inputs)  # will call loss_func(loss_mask, output_tensor) to get loss

    def train_step(self, data_list, train_info):
        '''
                RLHF calling
        rlhf framework source: self.train_step(train_data, train_info)


        :param data_list: what exactly? global batch? micro_batch?
        :param train_info:{"iteration": self.iteration} global outer iteration
        :return:
            None
        '''
        """Single training step."""
        iteration = train_info["iteration"]

        data_iterator = iter(data_list)
        _, skipped_iter, grad_norm, num_zeros_in_grad = megatron_train_step(self._forward_step, data_iterator,
                                                                            self.model, self.optimizer,
                                                                            self.opt_param_scheduler)
        self.post_update_stuffs({}, skipped_iter,
                                grad_norm, num_zeros_in_grad, iteration)

    def after_episode(self):

        '''
        RLHF calling
        :return:
        '''
        pass

    def before_episode(self):
        '''
        RLHF calling
        :return:
        '''
        pass

    def post_update_stuffs(self, loss_dict, skipped_iter,
                           grad_norm, num_zeros_in_grad, iteration):

        # only last rank give kl coef.
        if torch.distributed.get_rank() == (
            torch.distributed.get_world_size() - 1):
            self.kl_ctl.update(self.stats["policy/approx_kl"], n_steps=self.args.global_batch_size)

            self.put("kl_coef", self.kl_ctl.value)
            print(f"putting kl_coef ========================", self.kl_ctl.value, flush=True)

        # TODO tianhang get_num_microbatches scheduler is constants so it's fine for now. but if not we need 2 args
        self.args.consumed_train_samples += mpu.get_data_parallel_world_size() * \
                                            self.args.micro_batch_size * \
                                            get_num_microbatches()

        # Logging.
        loss_scale = self.optimizer.get_loss_scale().item()
        params_norm = None
        if self.args.log_params_norm:
            params_norm = calc_params_l2_norm(self.model)
        report_memory_flag = training_log(loss_dict, {},
                                          self.optimizer.param_groups[0]['lr'],
                                          iteration, loss_scale,
                                          self.report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad, self.stats, name="policy_trainer")
        self.report_memory_flag = report_memory_flag

# pylint: enable=unused-variable,invalid-name
