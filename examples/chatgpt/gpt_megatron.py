"""GPT"""


from functools import partial
import torch
import os
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.checkpointing import load_checkpoint
from megatron.model import GPTModel, ModelType
from megatron.training import get_model, build_train_valid_test_data_iterators, setup_model_and_optimizer
from megatron.training import train
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.text_generation.api import generate as megatron_generate
from megatron.initialize import initialize_megatron
from megatron.initialize import set_jit_fusion_options
from rlhf import RLHFTorchModule
import rlhf

# pylint: disable=unused-variable,invalid-name
class GPTMegatron(RLHFTorchModule):
    """gpt model wrapper"""


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = ModelType.encoder_or_decoder


    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        parallel_output = self.trainable
        model = GPTModel(
            num_tokentypes=0,
            parallel_output=parallel_output,
            pre_process=pre_process,
            post_process=post_process
        )
        return model


    def get_batch(self, data_iterator):
        """Generate a batch"""
        args = get_args()
        if args.dummy:
            bs = args.micro_batch_size
            sq = args.seq_length
            tokens = torch.ones([bs, sq], dtype=torch.int64, device='cuda')
            labels = torch.ones([bs, sq], dtype=torch.int64, device='cuda')
            position_ids = torch.ones([bs, sq], dtype=torch.int64, device='cuda')
            loss_mask = torch.ones([bs, sq], dtype=torch.float32, device='cuda')
            attention_mask = torch.ones([1, 1, sq, sq], dtype=torch.bool, device='cuda')
            return tokens, labels, loss_mask, attention_mask, position_ids

        tokenizer = get_tokenizer()

        # Items and their type.
        keys = ['text']
        datatype = torch.int64

        # Broadcast data.
        if data_iterator is not None:
            data = next(data_iterator)
        else:
            data = None
        if args.data_path is not None:
            data_b = mpu.broadcast_data(keys, data, datatype)

            # Unpack.
            tokens_ = data_b['text'].long()
            labels = tokens_[:, 1:].contiguous()
            tokens = tokens_[:, :-1].contiguous()

            # Get the masks and postition ids.
            attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                tokens,
                tokenizer.eod,
                args.reset_position_ids,
                args.reset_attention_mask,
                args.eod_mask_loss)

            return tokens, labels, loss_mask, attention_mask, position_ids
        else:
            # TODO(sayang): use dummy data
            assert data is not None
            bs = args.micro_batch_size
            sq = args.seq_length
            tokens = torch.ones([bs, sq], dtype=torch.int64, device='cuda')
            labels = torch.ones([bs, sq], dtype=torch.int64, device='cuda')
            position_ids = torch.ones([bs, sq], dtype=torch.int64, device='cuda')
            loss_mask = torch.ones([bs, sq], dtype=torch.float32, device='cuda')
            attention_mask = torch.ones([1, 1, sq, sq], dtype=torch.bool, device='cuda')
            return tokens, labels, loss_mask, attention_mask, position_ids


    def loss_func(self, loss_mask, output_tensor):
        args = get_args()
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        if args.average_loss_dp_mb:
            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])
        else:
            # todo(sayang): check for validation
            averaged_loss = [loss]

        return loss, {'lm loss': averaged_loss[0]}


    def forward_step(self, data_iterator, model):
        """Forward step."""
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator').start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        timers('batch-generator').stop()

        output_tensor = model(tokens, position_ids, attention_mask,
                              labels=labels)

        return output_tensor, partial(self.loss_func, loss_mask)


    def train_valid_test_dataset_provider(self, train_val_test_num_samples):
        """Build train, valid, and test datasets."""
        args = get_args()

        print_rank_0('> building train, validation, and test datasets '
                     'for GPT ...')
        if args.data_path is not None:
            train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
                data_prefix=args.data_path,
                data_impl=args.data_impl,
                splits_string=args.split,
                train_valid_test_num_samples=train_val_test_num_samples,
                seq_length=args.seq_length,
                seed=args.seed,
                skip_warmup=(not args.mmap_warmup))
            print_rank_0("> finished creating GPT datasets ...")
        else:
            # build data from shared storage
            # deal with train first, support valid and test later
            train_ds, valid_ds, test_ds = None, None, None
            train_ds = self.store.build_dataset(self.data_from, 1000)

        return train_ds, valid_ds, test_ds


    def setup(self):
        args_dict = self.model_args
        args_defaults = {'tokenizer_type': 'GPT2BPETokenizer'}
        assert args_dict is not None
        # Initalize and get arguments, timers, and Tensorboard writer.
        initialize_megatron(extra_args_provider=None,
                            ignore_unknown_args=True, # this must be True to be compitable with other system
                            args_defaults=args_defaults,
                            args_dict=args_dict)

        # Set pytorch JIT layer fusion options and warmup JIT functions.
        set_jit_fusion_options()

        args = get_args()
        timers = get_timers()

        # Model, optimizer, and learning rate.
        timers('model-and-optimizer-setup').start()
        self.model, self.optimizer, self.opt_param_scheduler = setup_model_and_optimizer(self.model_provider,
                                                                               self.model_type)
        timers('model-and-optimizer-setup').stop()

        # Data stuff.
        timers('train/valid/test-data-iterators-setup').start()
        if args.virtual_pipeline_model_parallel_size is not None:
            all_data_iterators = [
                build_train_valid_test_data_iterators(train_valid_test_dataset_provider)
                for _ in range(len(model))
            ]
            train_data_iterator = [data_iterators[0] for data_iterators in all_data_iterators]
            valid_data_iterator = [data_iterators[1] for data_iterators in all_data_iterators]
            test_data_iterator = [data_iterators[2] for data_iterators in all_data_iterators]
        else:
            self.train_data_iterator, self.valid_data_iterator, self.test_data_iterator \
                = build_train_valid_test_data_iterators(
                    self.train_valid_test_dataset_provider)
        timers('train/valid/test-data-iterators-setup').stop()

        # Print setup timing.
        print_rank_0('done with setup ...')
        return 1


    def train(self):
        args = get_args()
        print_rank_0('training ...')
        iteration = 0
        if args.do_train and args.train_iters > 0:
            iteration = train(self.forward_step,
                              self.model, self.optimizer, self.opt_param_scheduler,
                              self.train_data_iterator, self.valid_data_iterator,
                              process_non_loss_data_func=None)
        return 1


# pylint: enable=unused-variable,invalid-name
