"""Checkpoint Manager.
Code modified from https://code.alibaba-inc.com/algo/aimaster/blob/master/aimaster/python/torch/easyckpt/checkpoint_manager.py"""

import os
import pickle
import random
import shutil

import numpy as np
import torch
from rlhf.logger import log_rank_0
from itertools import cycle


class _DataLoader:
    """Data Loader Class."""

    def __init__(self, ckpt_mgr, loader, generator=None, is_cycle=False):
        self._ckpt_mgr = ckpt_mgr
        self._loader = loader
        self._generator = generator
        self._is_cycle = is_cycle
        self._data_loader_iter = None

    def __next__(self):
        next_data = next(self._data_loader_iter)
        return next_data

    def __iter__(self):
        self._data_loader_iter = self._ckpt_mgr._data_loader_iter(self._loader, self._generator, self._is_cycle)
        return self


def path_exists(path):
    return path and os.path.exists(path)


class CheckpointManager:
    """Checkpoint Mangaer."""

    def __init__(self, model, path, max_ckpt_nums, load_iteration=None):
        self.model = model
        self._path = path
        self._max_ckpt_nums = max_ckpt_nums
        if not path_exists(path):
            os.makedirs(path, exist_ok=True)
        self.load_iteration = load_iteration
        self._meta_file = os.path.join(self._path, "latest_checkpointed_iteration.txt")
        self.loader_index = 0
        self.loaders = {}
        self._objects = {'rng': {}, 'states': {}, 'tensor': {}}

    def _get_checkpoint_path_name(self, replica_id, step):
        name = "replica{}_step{}".format(replica_id, step)
        ckpt_path = os.path.join(self._path, name)
        return ckpt_path


    def _make_checkpoint_path(self, replica_id, step):
        ckpt_path = self._get_checkpoint_path_name(replica_id, step)
        if not path_exists(ckpt_path):
            os.mkdir(ckpt_path)
        return ckpt_path


    def _delete_ckpt_files(self):
        """Delete checkpoint files."""
        ckpt_path = self._path
        ckpt_files = os.listdir(ckpt_path)
        ckpt_folders = [(os.path.join(ckpt_path, f), os.path.getmtime(os.path.join(ckpt_path, f)))
                        for f in ckpt_files if os.path.isdir(os.path.join(ckpt_path, f))]
        ckpt_folders.sort(key=lambda x: x[1], reverse=True)
        ckpt_folders = [f[0] for f in ckpt_folders]
        reserved_folders = ckpt_folders[:self._max_ckpt_nums]
        for folder in ckpt_folders:
            if os.path.isdir(folder):
                if folder in reserved_folders:
                    continue
                try:
                    shutil.rmtree(folder)
                except PermissionError:
                    log_rank_0("Permission Denied: Please check the checkpoint file permissions.")

    def save_checkpoint(self, replica_id, train_iter, episode):
        ckpt_path = self._make_checkpoint_path(replica_id, train_iter)
        log_rank_0(f"save data checkpoint to {ckpt_path}, replica: {replica_id}, train_iter: {train_iter}, episode: {episode}")

        def _get_path(fn):
            return os.path.join(ckpt_path, fn)

        with open(_get_path('loaders.pkl'), 'wb') as f:
            if self.loaders[self.loader_index]['num_workers'] == 0:
                self._cache_dataset_rng_for_single_process()
            pickle.dump(self.loaders, f)
        with open(_get_path('obj.pkl'), 'wb') as f:
            for _, values in self._objects['rng'].items():
                values.append(values[0].get_rng_state())
            for _, values in self._objects['states'].items():
                values.append(values[0].state_dict())
            pickle.dump(self._objects, f)
        with open(_get_path("meta.pkl"), 'wb') as f:
            pickle.dump({"episode": episode, "train_iteration": train_iter}, f)

        self._set_latest_iteration(train_iter)
        # only reserve max nums of ckpt folders if needed
        if isinstance(self._max_ckpt_nums, int):
            self._delete_ckpt_files()
        log_rank_0("Checkpointing is done.")
        return True


    def resume_meta(self):
        ckpt_dir = self._get_checkpoint_path()
        if ckpt_dir is None:
            return
        with open(os.path.join(ckpt_dir, "meta.pkl"), 'rb') as f:
            return pickle.load(f)

    def _set_latest_iteration(self, iteration):
        with open(self._meta_file, 'w') as f:
            f.write(f"{iteration}")

    def _get_latest_iteration(self):
        if self.load_iteration is not None:
            return self.load_iteration
        if not path_exists(self._meta_file):
            return
        with open(self._meta_file) as f:
            iteration = f.read().strip()
            return iteration


    def _get_checkpoint_path(self):
        """Get checkpoint path."""
        latest_iter = self._get_latest_iteration()
        if latest_iter is None:
            log_rank_0(f"{self._meta_file} not found or load_iteration is not provided")
            return
        ckpt_path = self._get_checkpoint_path_name(self.model.replica_id, latest_iter)
        if path_exists(ckpt_path):
            log_rank_0(f"get checkpoint path from {self._path}")
            return ckpt_path
        log_rank_0(f"checkpoint path {ckpt_path} not exists")
        return


    def resume(self):
        """Resume data structures."""
        ckpt_path = self._get_checkpoint_path()
        if ckpt_path is None:
            log_rank_0("Do not have checkpoint files.")
        else:
            objects_path = os.path.join(ckpt_path, 'obj.pkl')
            if path_exists(objects_path):
                with open(objects_path, 'rb') as file:
                    resume_obj = pickle.load(file)
                    for key, values in resume_obj['rng'].items():
                        self._objects['rng'][key][0].set_rng_state(values[1])
                    for key, values in resume_obj['states'].items():
                        self._objects['states'][key][0].load_state_dict(values[1])
                    for key, values in resume_obj['tensor'].items():
                        self._objects['tensor'][key].copy_(values)

            loaders_path = os.path.join(ckpt_path, 'loaders.pkl')
            if path_exists(loaders_path):
                log_rank_0(f"resume from data checkpoint {loaders_path}")
                with open(loaders_path, 'rb') as file:
                    self.loaders = pickle.load(file)
                steps = [loader['step'] for _, loader in self.loaders.items()]

    def _init_states(self):
        loader_states = {'step': 0,
                         'num_workers': 0,
                         'random_state': None,
                         'cuda_random_state': None,
                         'base_seed': None,
                         'dataset_rng': None,
                         'cuda_dataset_rng': None}
        return loader_states

    def _cache_state(self, loader_states, generator=None):
        """Cache states for checkpoint manager."""
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        if generator is None:
            loader_states['random_state'] = [python_state, numpy_state, torch_state]
        else:
            genrator_state = generator.get_state()
            loader_states['random_state'] = [python_state, numpy_state, torch_state, genrator_state]
        if torch.cuda.is_available():
            loader_states['cuda_random_state'] = torch.cuda.get_rng_state()

    def _load_state(self, loader_states, generator=None):
        states = loader_states['random_state']
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if generator is not None:
            generator.set_state(states[3])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(loader_states['cuda_random_state'])

    def _cache_dataset_rng_for_single_process(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        self.loaders[self.loader_index]['dataset_rng'] = [python_state, numpy_state, torch_state]
        if torch.cuda.is_available():
            self.loaders[self.loader_index]['cuda_dataset_rng'] = torch.cuda.get_rng_state()

    def _load_dataset_rng_for_single_process(self, loader_states):
        if loader_states['dataset_rng']:
            states = loader_states['dataset_rng']
            random.setstate(states[0])
            np.random.set_state(states[1])
            torch.set_rng_state(states[2])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(loader_states['cuda_dataset_rng'])

    def _data_loader_iter(self, loader, generator=None, is_cycle=False):
        """Fast forward sampler and return data loader iter."""
        self.loader_index += 1
        # Triggerd when loaders are firtst resumed.
        if self.loader_index in self.loaders:
            loader_states = self.loaders[self.loader_index]
            self._load_state(loader_states, generator)
            data_loader_iter = loader.__iter__()
            data_loader_iter._base_seed = loader_states['base_seed']  # pylint: disable=protected-access

            if is_cycle:
                cycle_data_loader_iter = cycle(data_loader_iter)
            else:
                cycle_data_loader_iter = data_loader_iter
                
            if loader_states['step'] > 0:
                for _ in range(loader_states['step']):
                    next(cycle_data_loader_iter)
            if data_loader_iter._num_workers == 0:  # pylint: disable=protected-access
                self._load_dataset_rng_for_single_process(loader_states)
            return cycle_data_loader_iter
        loader_states = self._init_states()
        self._cache_state(loader_states, generator)
        data_loader_iter = loader.__iter__()
        if is_cycle:
            cycle_data_loader_iter = cycle(data_loader_iter)
        loader_states['num_workers'] = data_loader_iter._num_workers  # pylint: disable=protected-access
        loader_states['base_seed'] = data_loader_iter._base_seed  # pylint: disable=protected-access
        self.loaders[self.loader_index] = loader_states
        return cycle_data_loader_iter

    def data_loader(self, loader, generator=None, is_cycle=False):
        return _DataLoader(self, loader, generator, is_cycle)

    def add_step(self, step):
        self.loaders[self.loader_index]['step'] += step
        log_rank_0(f"loader index {self.loader_index} step {self.loaders[self.loader_index]['step']}")
