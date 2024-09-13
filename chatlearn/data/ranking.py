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
"""data ranking."""

def sort_fun(ele):
    chinese = ""
    others = ""
    for s in ele:
        if '\u4e00' <= s <= '\u9fa5':
            chinese += s
        else:
            others += s
    return len(others.split(" ")) + len(chinese)

def batch_generation_ranking(in_data, episode_per_epoch, sample_per_episode):
    for episode in range(episode_per_epoch):
        start = episode * sample_per_episode
        if episode < episode_per_epoch - 1:
            end = start + sample_per_episode
        else:
            end = len(in_data)
        cur_episode_sample = in_data[start:end]
        cur_episode_sample.sort(key=sort_fun, reverse=True)
        in_data = in_data[:start] + cur_episode_sample + in_data[end:]
    return in_data
