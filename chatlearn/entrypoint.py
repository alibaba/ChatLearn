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
"""chatlearn launcher"""

import argparse
from importlib import import_module
import sys
import traceback
from typing import Dict, Tuple, Type, Any
from omegaconf import OmegaConf
from transformers import AutoConfig
from omegaconf.dictconfig import DictConfig
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore

from chatlearn.algorithm.base_algo import BaseAlgorithm


# e.g. python chatlearn/chatlearn.py grpo --config-file grpo.yaml runtime.data_path=/tmp/data runtime.eval_data_path=/tmp/eval_data

# Registry format:
#  "engine_name": ("module_path", "algo_class_name", "config_class")
ALGO_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "grpo": ("algorithm.grpo", "GrpoAlgorithm", "GrpoConfig"),
}


class ChatlearnLauncher:
    """ChatlearnLauncher"""

    def __init__(self) -> None:
        self.parser = self._create_parser()


    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="ChatLearn: An RLHF Training System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # add_help=False,
        )

        subparsers = parser.add_subparsers(
            title="Available algorithms",
            dest="algorithm",
            metavar="ALGORITHM"
        )

        for algo_name in ALGO_REGISTRY:
            algo_parser = subparsers.add_parser(
                algo_name,
                description=f"Run {algo_name.upper()} algorithm",
                help=f"{algo_name.upper()} algorithm",
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            algo_parser.add_argument(
                "--config-file",
                type=str,
                help="Path to the config file",
            )
            algo_parser.add_argument(
                "hydra_args",
                nargs=argparse.REMAINDER,
                help="Hydra configs (e.g. ++key=value)"
            )

        return parser


    def _load_algorithm(self, algo_name: str) -> Tuple[Type[BaseAlgorithm], Type[Any]]:
        module_path, algo_cls_name, config_cls_name = ALGO_REGISTRY[algo_name]
        try:
            module = import_module(module_path)
            algo_cls = getattr(module, algo_cls_name)
            config_cls = getattr(module, config_cls_name)
            return algo_cls, config_cls
        except Exception as e:
            raise RuntimeError(f"Failed to load algorithm: {algo_name} ({str(e)})") from e

    def _update_external_config(self, hf_transformer_config: AutoConfig, external_cfg: DictConfig) -> DictConfig:
        # external_cfg['policy_trainer']['megatron_model_cfg']
        # transformers.models.qwen3.configuration_qwen3.Qwen3Config
        # omegaconf.dictconfig.DictConfig
        megatron_model_cfg = {}
        megatron_model_cfg['num_layers'] = hf_transformer_config.num_hidden_layers
        megatron_model_cfg['hidden_size'] = hf_transformer_config.hidden_size
        megatron_model_cfg['num_attention_heads'] = hf_transformer_config.num_attention_heads
        megatron_model_cfg['ffn_hidden_size'] = hf_transformer_config.intermediate_size
        megatron_model_cfg['num_query_groups'] = hf_transformer_config.num_key_value_heads
        megatron_model_cfg['max_position_embeddings'] = hf_transformer_config.max_position_embeddings
        megatron_model_cfg['add_qkv_bias'] = hf_transformer_config.hidden_size
        megatron_model_cfg['add_bias_linear'] = hf_transformer_config.hidden_size
        megatron_model_cfg['rotary_base'] = hf_transformer_config.rope_theta
        megatron_model_cfg['norm_epsilon'] = hf_transformer_config.rms_norm_eps
        megatron_model_cfg['group_query_attention'] = True
        megatron_model_cfg['untie_embeddings_and_output_weights'] = not hf_transformer_config.tie_word_embeddings
        megatron_model_cfg['group_query_attention'] = hf_transformer_config.hidden_size
        megatron_model_cfg['untie_embeddings_and_output_weights'] = hf_transformer_config.hidden_size
        megatron_model_cfg['vocab_size'] = hf_transformer_config.vocab_size

        OmegaConf.update(external_cfg['models']['policy_trainer'], 'megatron_model_cfg', megatron_model_cfg)
        
        return external_cfg

    def _run_algorithm(self, algo_args) -> None:
        algo_cls, config_cls = self._load_algorithm(algo_args.algorithm)
        cs = ConfigStore.instance()
        cs.store(name=algo_args.algorithm, node=config_cls)
        GlobalHydra.instance().clear()
        with hydra.initialize(config_path=None, version_base=None):
            cfg = hydra.compose(config_name=algo_args.algorithm)
            if algo_args.config_file is not None:
                external_cfg = OmegaConf.load(algo_args.config_file)
                hf_model_path = None
                train_backend = None 
                for arg in algo_args.hydra_args:
                    if '=' in arg:
                        key, value = arg.split('=', 1)
                        origin_type = type(OmegaConf.select(external_cfg, key))
                        if origin_type == bool:
                            OmegaConf.update(external_cfg, key, value.lower().strip() == 'true')
                        else:
                            OmegaConf.update(external_cfg, key, origin_type(value))
                    if 'models.policy.load' in arg:
                        hf_model_path = arg.split('=', 1)[1]
                    if 'train_backend' in arg:
                        train_backend = arg.split('=', 1)[1]
                if train_backend == "megatron":
                    hf_transformer_config = AutoConfig.from_pretrained(hf_model_path)
                    external_cfg = self._update_external_config(hf_transformer_config, external_cfg)
                cfg = OmegaConf.merge(cfg, external_cfg) # include $
            cfg = OmegaConf.to_object(cfg) # real cfg
            instance = algo_cls(cfg)
            instance.run()


    def run(self) -> None:
        args, _ = self.parser.parse_known_args()
        
        if not args.algorithm:
            self.parser.print_help()
            return

        if args.algorithm not in ALGO_REGISTRY:
            print(f"ERROR: Unknown algorithm {args.algorithm}")
            self.parser.print_help()
            sys.exit(1)

        algo_args = self.parser.parse_args()
 
        try:
            self._run_algorithm(algo_args)
        except Exception:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    launcher = ChatlearnLauncher()
    launcher.run()
