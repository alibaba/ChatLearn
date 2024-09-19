# Code below is modified from NVIDIA Megatron-LM
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
"""Convertion script"""

import argparse
import importlib
import sys
import torch.multiprocessing as mp


# Code below is copied from Megatron-LM core v0.8.0
def load_plugin(plugin_type, name):
    module_name = f"{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError as e1:
        print(e1)
        module_name = name
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError as e2:
            print(e2)
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    print(f"Loaded {module_name} as the {plugin_type}.")
    return plugin

def main():
    parser = argparse.ArgumentParser(description="Megatron Checkpoint Converter Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--loader-prefix', type=str, default='loader',
                        help='Prefix import path for loader')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, should be on python path')
    parser.add_argument('--saver-prefix', type=str, default='saver',
                        help='Prefix import path for saver')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')

    known_args, _ = parser.parse_known_args()
    loader = load_plugin(known_args.loader_prefix, known_args.loader)
    saver = load_plugin(known_args.saver_prefix, known_args.saver)

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=args.max_queue_size)

    print("Starting saver...")
    saver_proc = mp.Process(target=saver.save_checkpoint, args=(queue, args))
    saver_proc.start()

    print("Starting loader...")
    loader.load_checkpoint(queue, args)

    print("Waiting for saver to complete...")
    saver_proc.join()


if __name__ == '__main__':
    main()
