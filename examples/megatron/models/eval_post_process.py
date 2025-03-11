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
"""evaluator post process"""

import numpy as np
import torch

from chatlearn.utils.utils import listdict_to_dictlist
from .utils import write_jsonl


class EvaluatorPostProcess(Evaluator):
    """Evaluator post-process"""

    def post_process(self, results, eval_info):
        """
        Default post-process function for model evaluation results.

        Args
        ----
            results: list[]
                a list of evaluation results
            eval_info: dict[]
                a meta that contains "train_iteration" and "episode_iteration"
        """
        args = self.args
        results = results["reward"]
        results = listdict_to_dictlist(results)
        if args.get('eval_data_num_limit') > 0:
            assert len(results['rewards']) == args.get('eval_data_num_limit'), (
                f"expect {len(results['rewards'])} == {args.get('eval_data_num_limit')}"
            )

        eval_reward_stats = {"eval_reward_mean": np.mean(results['rewards'])}
        train_iteration = eval_info["train_iteration"]

        logger.info(f"eval reward stats: {eval_reward_stats} iter: {train_iteration}")

        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                self._metric_list.append(eval_reward_stats)
        else:
            self._metric_list.append(eval_reward_stats)

        print(f"eval reward stats: {eval_reward_stats} iter: {train_iteration}")
        save_fp = f"{args.output_dir}/eval/{train_iteration}/eval_json_res.json" # pylint: disable=line-too-long
        write_jsonl(results["eval_jsonl"], save_fp)

        return results
