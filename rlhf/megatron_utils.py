import re
import functools


# regex to parse out layer number from param name
layer_re = re.compile('layers\.([0-9]+)')


def update_layer_num(layers_per_part, rank, m):
    # This assumes no interleaved pipeline execution
    layer = int(m.group(1))
    layer += rank * layers_per_part
    return f'layers.{layer}'


def build_pipeline_layer_name_mapping(layers_per_stage, rank, model):
    name_mapping = {}
    for src_name, partition_param in model.named_parameters():
        if src_name.endswith("word_embeddings.weight") and "language_model" not in src_name:
            # See comment in MegatronModule.initialize_word_embeddings()
            tgt_name = src_name.replace("word_embeddings.weight", "language_model.embedding.word_embeddings.weight")
        else:
            # Translate destination layer number (0-N for each partition)
            # to source layer number (single-model layer number)
            _update_layer_num = functools.partial(update_layer_num, layers_per_stage, rank)
            tgt_name = re.sub(layer_re, _update_layer_num, src_name)
        name_mapping[tgt_name] = src_name
    return name_mapping
