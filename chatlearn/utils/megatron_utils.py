# Copyright 2023 Alibaba Group Holding Limited. All Rights Reserved.
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
"""megatron utils"""

import functools
import re

# regex to parse out layer number from param name
layer_re = re.compile(r'layers\.([0-9]+)')


def update_layer_num(layers_per_part, rank, m):
    # This assumes no interleaved pipeline execution
    layer = int(m.group(1))
    layer += rank * layers_per_part
    return f'layers.{layer}'


def build_pipeline_layer_name_mapping(src_layers_per_stage, src_rank, map_interval, tgt_last_stage, model, requires_grad):
    """
    remap pipeline layer_name. For each pipeline stage, the layer number starts with 0.
    Args:
        src_layers_per_stage: layer_per_stage in src model
        src_rank: src model pipeline rank
        map_interval: map interval from tgt to src, i.e. if src_layers_per_stage is 2, and tgt_layers_per_stage is 4,
                      then the map_iterval is tgt_layers_per_stage/src_layers_per_stage = 2
        tgt_last_stage: is target model in last stage
        model: megatron model
        requires_grad: whether the layer requires grad
    """
    name_mapping = {}
    for src_name, partition_param in model.named_parameters():
        if requires_grad:
            if not partition_param.requires_grad:
                continue
        if src_name.endswith("word_embeddings.weight") and "language_model" not in src_name:
            # See comment in MegatronModule.initialize_word_embeddings()
            if not tgt_last_stage:
                tgt_name = src_name.replace("word_embeddings.weight", "language_model.embedding.word_embeddings.weight")
            else:
                tgt_name = src_name
        else:
            # Translate destination layer number (0-N for each partition)
            # to source layer number (single-model layer number)
            rank = src_rank % map_interval
            _update_layer_num = functools.partial(update_layer_num, src_layers_per_stage, rank)
            tgt_name = re.sub(layer_re, _update_layer_num, src_name)
        name_mapping[tgt_name] = src_name
    return name_mapping
