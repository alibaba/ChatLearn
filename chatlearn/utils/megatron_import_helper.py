# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""""Version compatibility for megatron"""

# pylint: disable=unused-import

# megatron.*
try:
    from megatron import arguments
    from megatron import get_args
    from megatron import get_num_microbatches
    from megatron import get_timers
    from megatron import get_tokenizer
    from megatron import is_last_rank
    from megatron import print_rank_0
    from megatron import print_rank_last
    from megatron import update_num_microbatches
except ImportError:
    from megatron.training import arguments
    from megatron.training import get_args
    from megatron.training import get_num_microbatches
    from megatron.training import get_timers
    from megatron.training import get_tokenizer
    from megatron.training import is_last_rank
    from megatron.training import print_rank_0
    from megatron.training import print_rank_last
    from megatron.training import update_num_microbatches

# megatron.arguments.*
try:
    from megatron.arguments import parse_args
    from megatron.arguments import validate_args
except ImportError:
    from megatron.training.arguments import parse_args
    from megatron.training.arguments import validate_args

# megatron.checkpointing.*
try:
    from megatron.checkpointing import _load_base_checkpoint
    from megatron.checkpointing import load_args_from_checkpoint
    from megatron.checkpointing import load_checkpoint
    from megatron.checkpointing import find_checkpoint_rank_0
    from megatron.checkpointing import fix_query_key_value_ordering
    from megatron.checkpointing import get_checkpoint_tracker_filename
    from megatron.checkpointing import get_checkpoint_version
    from megatron.checkpointing import set_checkpoint_version
    from megatron.checkpointing import read_metadata
except ImportError:
    from megatron.training.checkpointing import _load_base_checkpoint
    from megatron.training.checkpointing import load_args_from_checkpoint
    from megatron.training.checkpointing import load_checkpoint
    from megatron.training.checkpointing import find_checkpoint_rank_0
    from megatron.training.checkpointing import fix_query_key_value_ordering
    from megatron.training.checkpointing import get_checkpoint_tracker_filename
    from megatron.training.checkpointing import get_checkpoint_version
    from megatron.training.checkpointing import set_checkpoint_version
    from megatron.training.checkpointing import read_metadata

# megatron.global_vars.*
try:
    from megatron.global_vars import get_tensorboard_writer
    from megatron.global_vars import set_global_variables
except ImportError:
    from megatron.training.global_vars import get_tensorboard_writer
    from megatron.training.global_vars import set_global_variables

# megatron.initialize.*
try:
    from megatron.initialize import _initialize_distributed
    from megatron.initialize import _set_random_seed
    from megatron.initialize import _init_autoresume
    from megatron.initialize import _compile_dependencies
    from megatron.initialize import initialize_megatron
    from megatron.initialize import set_jit_fusion_options
except ImportError:
    from megatron.training.initialize import _initialize_distributed
    from megatron.training.initialize import _set_random_seed
    from megatron.training.initialize import _init_autoresume
    from megatron.training.initialize import _compile_dependencies
    from megatron.training.initialize import initialize_megatron
    from megatron.training.initialize import set_jit_fusion_options

# megatron.training.*
try:
    from megatron.training import get_optimizer_param_scheduler
    from megatron.training import print_datetime
    from megatron.training import train_step
except ImportError:
    from megatron.training.training import get_optimizer_param_scheduler
    from megatron.training.training import print_datetime
    from megatron.training.training import train_step

# megatron.utils.*
try:
    from megatron.utils import average_losses_across_data_parallel_group
    from megatron.utils import calc_params_l2_norm
    from megatron.utils import get_ltor_masks_and_position_ids
    from megatron.utils import unwrap_model
except ImportError:
    from megatron.training.utils import average_losses_across_data_parallel_group
    from megatron.training.utils import calc_params_l2_norm
    from megatron.training.utils import get_ltor_masks_and_position_ids
    from megatron.training.utils import unwrap_model

# megatron.model.*
try:
    from megatron.model import GPTModel
    from megatron.model.language_model import parallel_lm_logits
    from megatron.model.module import MegatronModule
    from megatron.model.utils import get_linear_layer
    from megatron.model.enums import AttnType
    from megatron.model import Float16Module
except ImportError:
    from megatron.legacy.model import GPTModel
    from megatron.legacy.model.language_model import parallel_lm_logits
    from megatron.legacy.model.module import MegatronModule
    from megatron.legacy.model.utils import get_linear_layer
    from megatron.legacy.model.enums import AttnType
    from megatron.legacy.model import Float16Module

# megatron.text_generation.*
try:
    from megatron.text_generation import generation
    from megatron.text_generation.communication import broadcast_float_list
    from megatron.text_generation.communication import broadcast_int_list
    from megatron.text_generation.communication import broadcast_tensor
    from megatron.text_generation.communication import send_to_next_pipeline_rank
    from megatron.text_generation.communication import recv_from_prev_pipeline_rank_
    from megatron.text_generation.forward_step import _allocate_recv_buffer
    from megatron.text_generation.generation import generate_tokens_probs_and_return_on_first_stage
except ImportError:
    from megatron.inference.text_generation import generation
    from megatron.inference.text_generation.communication import broadcast_float_list
    from megatron.inference.text_generation.communication import broadcast_int_list
    from megatron.inference.text_generation.communication import broadcast_tensor
    from megatron.inference.text_generation.communication import send_to_next_pipeline_rank
    from megatron.inference.text_generation.communication import recv_from_prev_pipeline_rank_
    from megatron.inference.text_generation.forward_step import _allocate_recv_buffer
    from megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage

# megatron.optimizer.*
try:
    from megatron.optimizer import get_megatron_optimizer
    from megatron.optimizer import DistributedOptimizer
    from megatron.optimizer.optimizer import MegatronOptimizer
    from megatron.optimizer.optimizer import MixedPrecisionOptimizer
    from megatron.optimizer.optimizer import Float16OptimizerWithFloat16Params
except ImportError:
    from megatron.core.optimizer import get_megatron_optimizer
    from megatron.core.optimizer import DistributedOptimizer
    from megatron.core.optimizer.optimizer import MegatronOptimizer
    from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
    from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params

# DistributedDataParallel
try:
    from megatron.core import DistributedDataParallel
except ImportError:
    try:
        from megatron.core.distributed import DistributedDataParallel
    except ImportError:
        from megatron.model.distributed import DistributedDataParallel


# megatron.core.*
from megatron.core import mpu
from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.pipeline_parallel import schedules
from megatron.core.tensor_parallel.utils import VocabUtility

from megatron.core.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_global_memory_buffer
)

from megatron.core.tensor_parallel.layers import (
    ColumnParallelLinear,
    linear_with_grad_accumulation_and_async_allreduce,
    LinearWithGradAccumulationAndAsyncCommunication,
    RowParallelLinear,
    VocabParallelEmbedding
)

try:
    from megatron.core.tensor_parallel.layers import linear_with_frozen_weight
except ImportError:
    linear_with_frozen_weight = None

from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region
)

# pylint: enable=unused-import


def save_checkpoint_and_time(iteration, model, optimizer, opt_param_scheduler):
    try:
        from megatron.training import save_checkpoint_and_time as save_checkpoint_and_time_v1 # pylint: disable=import-outside-toplevel
        save_checkpoint_and_time_v1(iteration, model, optimizer, opt_param_scheduler)
    except ImportError:
        from megatron.training.training import save_checkpoint_and_time as save_checkpoint_and_time_v2# pylint: disable=import-outside-toplevel
        save_checkpoint_and_time_v2(iteration, model, optimizer, opt_param_scheduler, 0, None)
