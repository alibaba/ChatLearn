import time

import torch


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.start_time = time.time()
        self._num = 0

    def start(self):
        """Start the timer."""
        assert not self.started_, f'timer {self.name_} has already been started'
        torch.cuda.synchronize()
        self.start_time = time.time()
        self.started_ = True
        self._num += 1

    def stop(self):
        """Stop the timer."""
        torch.cuda.synchronize()
        self.elapsed_ += (time.time() - self.start_time)
        self.started_ = False

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self._num = 0

    def elapsed(self, reset=True, return_num=False):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        num_ = self._num
        # Reset the elapsed time
        if reset:
            self.reset()
        # If timing was in progress, set it back.
        if started_:
            self.start()
        if return_num:
            return elapsed_, num_
        return elapsed_


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        # currently when using add_scalars,
        # torch.utils.add_scalars makes each timer its own run, which
        # polutes the runs list, so we just add each as a scalar
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, names=None, normalizer=1.0, reset=True, return_dict=False):
        """Log a group of timers."""
        if names is None:
            names = self.timers.keys()
        name2log = {}
        assert normalizer > 0.0
        string = 'time (ms)'
        for name in names:
            elapsed_time, num = self.timers[name].elapsed(reset=reset, return_num=True)
            elapsed_time = elapsed_time * 1000.0 / normalizer

            if num >= 1:
                avg_elapsed_time = elapsed_time / num
                string += ' | {}: {:.2f}(avg: {:.2f})'.format(name, elapsed_time, avg_elapsed_time)
            else:
                string += ' | {}: {:.2f}'.format(name, elapsed_time)
            if return_dict:
                name2log[name] = elapsed_time
        if return_dict:
            return string, name2log
        else:
            return string
