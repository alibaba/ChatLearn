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
"""Checkpoint Manager."""

import os
import pickle
import shutil

from chatlearn.utils.logger import log_rank_0

def path_exists(path):
    """path exits"""
    return path and os.path.exists(path)

class CheckpointManager:
    """Checkpoint Manager"""

    def __init__(self, model, path, max_ckpt_nums, load_iteration=None, config_to_check=None):
        self._path = path
        self._max_ckpt_nums = max_ckpt_nums
        self._meta_file = os.path.join(self._path, "latest_checkpointed_iteration.txt")
        if not path_exists(path):
            os.makedirs(path, exist_ok=True)
        self.load_iteration = load_iteration
        self._meta = None
        self._model = model
        self._resumed = False
        self._config_to_check = {} if config_to_check is None else config_to_check


    def _get_checkpoint_path_name(self, replica_id, step):
        name = "replica{}_step{}".format(replica_id, step)
        ckpt_path = os.path.join(self._path, name)
        return ckpt_path

    def _make_checkpoint_path(self, replica_id, step):
        ckpt_path = self._get_checkpoint_path_name(replica_id, step)
        if not path_exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
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

    def save_checkpoint(self, replica_id, train_iter, episode, consumed_samples):
        """save data checkpoint"""
        ckpt_path = self._make_checkpoint_path(replica_id, train_iter)
        log_rank_0(
            f"save data checkpoint to {ckpt_path}, replica: {replica_id}, train_iter: {train_iter}, episode: {episode} " + \
            f"consumed samples {consumed_samples}")

        def _get_path(fn):
            return os.path.join(ckpt_path, fn)

        meta_data = {"episode": episode,
                     "train_iteration": train_iter,
                     "consumed_samples": consumed_samples,
                     "sample_per_episode": self._model.runtime_args.sample_per_episode,
                     "data_ratio": self._model.runtime_args.data_ratio}

        with open(_get_path("meta.pkl"), 'wb') as f:
            pickle.dump(meta_data, f)
        if replica_id == 0:
            self._set_latest_iteration(train_iter)
            # only reserve max nums of ckpt folders if needed
            if isinstance(self._max_ckpt_nums, int):
                self._delete_ckpt_files()
        log_rank_0("Checkpointing is done.")
        return True

    def _set_latest_iteration(self, iteration):
        with open(self._meta_file, 'w', encoding='utf-8') as f:
            f.write(f"{iteration}")

    def _get_checkpoint_path(self):
        """Get checkpoint path."""
        latest_iter = self._get_latest_iteration()
        if latest_iter is None:
            log_rank_0(f"{self._meta_file} not found or load_iteration is not provided")
            return
        ckpt_path = self._get_checkpoint_path_name(self._model.replica_id, latest_iter)
        if path_exists(ckpt_path):
            log_rank_0(f"get checkpoint path from {self._path}")
            return ckpt_path
        log_rank_0(f"checkpoint path {ckpt_path} not exists")
        return

    def validate(self, ckpt_meta):
        for key, value in self._config_to_check.items():
            assert value == ckpt_meta[key], \
                f"config {key}: {value} diff with ckpt config {ckpt_meta[key]}"

    def resume_meta(self):
        if self._meta is not None:
            return self._meta
        ckpt_dir = self._get_checkpoint_path()
        if ckpt_dir is None:
            return
        with open(os.path.join(ckpt_dir, "meta.pkl"), 'rb') as f:
            self._meta = pickle.load(f)
        self.validate(self._meta)
        return self._meta

    def resume(self):
        """Resume data structures."""
        if self._resumed:
            return self._meta
        meta = self.resume_meta()
        if meta is not None:
            self._model.runtime_args.consumed_samples = meta["consumed_samples"]
            log_rank_0(f"set consumed_samples to {meta['consumed_samples']}")
            self._model.runtime_args.data_ratio = data_ratio = meta.get("data_ratio", None)
            log_rank_0(f"set data_ratio to {data_ratio}")
        self._resumed = True
        return meta

    def _get_latest_iteration(self):
        if self.load_iteration is not None:
            return self.load_iteration
        if not path_exists(self._meta_file):
            return
        with open(self._meta_file, encoding='utf-8') as f:
            iteration = f.read().strip()
            return iteration
