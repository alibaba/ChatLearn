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
"""MathRuleRM"""

import timeout_decorator
from tqdm import tqdm

from .math_utils.grader import math_equal_process
from .math_utils.parser import extract_answer_custom, strip_string


@timeout_decorator.timeout(3)
def math_equal_timeout(param):
    return math_equal_process(param)


class MathRuleRM:
    """math rule reward model"""

    def __init__(
            self,
            timeout=1,
    ):
        self.timeout = timeout

    def __call__(self, data):
        params = []
        for idx, (answer, str_output) in enumerate(data):
            pred = extract_answer_custom(
                str_output,
                use_last_number=True,
                use_choice=False
            )
            pred = strip_string(pred)
            params.append([idx, pred, answer])

        scores = []
        pbar = tqdm(total=len(params))
        for param in params:
            try:
                result = math_equal_timeout(param)
            except timeout_decorator.timeout_decorator.TimeoutError:
                result = False
            scores.append(result)
            pbar.update(1)

        rewards = [0 for _ in range(len(data))]
        extract_success = [1 for _ in range(len(data))]
        for (idx, pred, answer), score in zip(params, scores):
            rewards[idx] = float(score)
            if not pred:
                extract_success[idx] = 0
        return rewards, extract_success
