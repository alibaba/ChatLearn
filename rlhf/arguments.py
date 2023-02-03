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
"""arguments from command or yaml."""

import argparse
import yaml


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='RLHF Arguments',
                                     allow_abbrev=False)

    parser.add_argument("-c", "--config",
                        default=None,
                        help="where to load YAML configuration",
                        metavar="FILE")
    parser.add_argument("-p", "--shared-path",
                        default=None,
                        help="where to store shared information",
                        type=str)

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r', encoding='utf-8') as stream:
            config_vars = yaml.load(stream, Loader=yaml.FullLoader)
            for key, value in config_vars.items():
                setattr(args, key, value)

    return args
