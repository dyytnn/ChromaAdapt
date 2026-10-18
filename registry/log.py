import logging
import os
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from abs_interface import Abstract_SegModel
from utils import is_main_process

_CURRENT_STORAGE_STACK = []

def setup_logger(log_dir="logs", logger_name="EpochStatsLogger", log_prefix="train"):
    log_filename = f"{log_prefix}_{datetime.now().strftime('%Y-%m-%d')}.log"
    log_path = os.path.join(log_dir, log_filename)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 🔒 Check if a FileHandler for this file already exists
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_path)
               for h in logger.handlers):
        file_handler = logging.FileHandler(log_path, mode='a')  # 'a' = append mode
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger

def _fmt(x: float) -> str:
    ax = abs(x)
    # use sci for |x| < 1e-3 or |x| >= 1e+4, else fixed
    return f"{x:.3e}" if (ax != 0 and (ax < 1e-3 or ax >= 1e4)) else f"{x:.6f}"

class StatAccumulator:
    def __init__(self):
        self.data = defaultdict(list)
    def add(self, name: str, tensor: Tensor):
        if not torch.is_tensor(tensor): return
        flat = tensor.reshape(-1).float()
        self.data[f"{name}_norm2"].append(float(torch.linalg.vector_norm(flat)))
        self.data[f"{name}_absmax"].append(float(flat.abs().max()))
        self.data[f"{name}_mean"].append(float(flat.mean()))


    def log_epoch_summary(self, logger, prefix=""):
        for key, values in self.data.items():
            if values:
                mean_val = sum(values) / len(values)
                logger.info(f"{prefix}{key}: {_fmt(mean_val)}")
        self.data.clear()


def enable_epoch_logging(model: Abstract_SegModel, accumulator: StatAccumulator, log_input=True, log_grad=False):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Module) and not isinstance(module, torch.nn.Sequential) and is_main_process():
            def forward_hook(module, input, output, name=name):
                if log_input and isinstance(input, tuple):
                    accumulator.add(f"[Input] {name}", input[0])
                if log_input and isinstance(output, torch.Tensor):
                    accumulator.add(f"[Output] {name}", output)

            def backward_hook(module, grad_input, grad_output, name=name):
                if log_grad and isinstance(grad_output[0], torch.Tensor):
                    logged_grad_output = grad_output[0].clone()
                    accumulator.add(f"[Grad Output] {name}", logged_grad_output)

            if log_input:
                module.register_forward_hook(forward_hook)
            if log_grad:
                module.register_full_backward_hook(backward_hook)

class HistoryBuffer:
    """
    Track a series of scalar values and provide access to smoothed values over a
    window or the global average of the series.
    """

    def __init__(self, max_length: int = 1000000) -> None:
        """
        Args:
            max_length: maximal number of values that can be stored in the
                buffer. When the capacity of the buffer is exhausted, old
                values will be removed.
        """
        self._max_length: int = max_length
        self._data: List[Tuple[float, float]] = []  # (value, iteration) pairs
        self._count: int = 0
        self._global_avg: float = 0

    def update(self, value: float, iteration: Optional[float] = None) -> None:
        """
        Add a new scalar value produced at certain iteration. If the length
        of the buffer exceeds self._max_length, the oldest element will be
        removed from the buffer.
        """
        if iteration is None:
            iteration = self._count
        if len(self._data) == self._max_length:
            self._data.pop(0)
        self._data.append((value, iteration))

        self._count += 1
        self._global_avg += (value - self._global_avg) / self._count

    def latest(self) -> float:
        """
        Return the latest scalar value added to the buffer.
        """
        return self._data[-1][0]

    def median(self, window_size: int) -> float:
        """
        Return the median of the latest `window_size` values in the buffer.
        """
        return np.median([x[0] for x in self._data[-window_size:]])

    def avg(self, window_size: int) -> float:
        """
        Return the mean of the latest `window_size` values in the buffer.
        """
        return np.mean([x[0] for x in self._data[-window_size:]])

    def global_avg(self) -> float:
        """
        Return the mean of all the elements in the buffer. Note that this
        includes those getting removed due to limited buffer storage.
        """
        return self._global_avg

    def values(self) -> List[Tuple[float, float]]:
        """
        Returns:
            list[(number, iteration)]: content of the current buffer.
        """
        return self._data

class EventStorage:
    """
        The user-facing class that provides metrics storage and logging functionalities.
    """
    def __init__(self, start_iter=0):
        """
        Args:
            start_iter (int): the iteration number to start with
        """
        self._history = defaultdict(HistoryBuffer)
        self._smoothing_hints = {}
        self._latest_scalars = {}
        self._iter: int = start_iter
        self._current_prefix = ""
        self._vis_data = []
        self._histograms = []

    def put_image(self, img_name, img_tensor):
        """
        Add an `img_tensor` associated with `img_name`, to be shown on
        tensorboard.

        Args:
            img_name (str): The name of the image to put into tensorboard.
            img_tensor (torch.Tensor or numpy.array): An `uint8` or `float`
                Tensor of shape `[channel, height, width]` where `channel` is
                3. The image format should be RGB. The elements in img_tensor
                can either have values in [0, 1] (float32) or [0, 255] (uint8).
                The `img_tensor` will be visualized in tensorboard.
        """
        self._vis_data.append((img_name, img_tensor, self._iter))

    def put_scalar(self, name, value, smoothing_hint=True, cur_iter=None):
        """
        Add a scalar `value` to the `HistoryBuffer` associated with `name`.

        Args:
            smoothing_hint (bool): a 'hint' on whether this scalar is noisy and should be
                smoothed when logged. The hint will be accessible through
                :meth:`EventStorage.smoothing_hints`.  A writer may ignore the hint
                and apply custom smoothing rule.

                It defaults to True because most scalars we save need to be smoothed to
                provide any useful signal.
            cur_iter (int): an iteration number to set explicitly instead of current iteration
        """
        name = self._current_prefix + name
        cur_iter = self._iter if cur_iter is None else cur_iter
        history = self._history[name]
        value = float(value)
        history.update(value, cur_iter)
        self._latest_scalars[name] = (value, cur_iter)

        existing_hint = self._smoothing_hints.get(name)

        if existing_hint is not None:
            assert (
                existing_hint == smoothing_hint
            ), "Scalar {} was put with a different smoothing_hint!".format(name)
        else:
            self._smoothing_hints[name] = smoothing_hint
    
    def put_scalars(self, *, smoothing_hint=True, cur_iter=None, **kwargs):
        """
        Put multiple scalars from keyword arguments.

        Examples:

            storage.put_scalars(loss=my_loss, accuracy=my_accuracy, smoothing_hint=True)
        """
        for k, v in kwargs.items():
            self.put_scalar(k, v, smoothing_hint=smoothing_hint, cur_iter=cur_iter)

    def put_histogram(self, hist_name, hist_tensor, bins=1000):
        """
        Create a histogram from a tensor.

        Args:
            hist_name (str): The name of the histogram to put into tensorboard.
            hist_tensor (torch.Tensor): A Tensor of arbitrary shape to be converted
                into a histogram.
            bins (int): Number of histogram bins.
        """
        ht_min, ht_max = hist_tensor.min().item(), hist_tensor.max().item()

        # Create a histogram with PyTorch
        hist_counts = torch.histc(hist_tensor, bins=bins)
        hist_edges = torch.linspace(start=ht_min, end=ht_max, steps=bins + 1, dtype=torch.float32)

        # Parameter for the add_histogram_raw function of SummaryWriter
        hist_params = dict(
            tag=hist_name,
            min=ht_min,
            max=ht_max,
            num=len(hist_tensor),
            sum=float(hist_tensor.sum()),
            sum_squares=float(torch.sum(hist_tensor**2)),
            bucket_limits=hist_edges[1:].tolist(),
            bucket_counts=hist_counts.tolist(),
            global_step=self._iter,
        )
        self._histograms.append(hist_params)

    def history(self, name):
        """
        Returns:
            HistoryBuffer: the scalar history for name
        """
        ret = self._history.get(name, None)
        if ret is None:
            raise KeyError("No history metric available for {}!".format(name))
        return ret

    def histories(self):
        """
        Returns:
            dict[name -> HistoryBuffer]: the HistoryBuffer for all scalars
        """
        return self._history

    def latest(self):
        """
        Returns:
            dict[str -> (float, int)]: mapping from the name of each scalar to the most
                recent value and the iteration number its added.
        """
        return self._latest_scalars

    def latest_with_smoothing_hint(self, window_size=20):
        """
        Similar to :meth:`latest`, but the returned values
        are either the un-smoothed original latest value,
        or a median of the given window_size,
        depend on whether the smoothing_hint is True.

        This provides a default behavior that other writers can use.

        Note: All scalars saved in the past `window_size` iterations are used for smoothing.
        This is different from the `window_size` definition in HistoryBuffer.
        Use :meth:`get_history_window_size` to get the `window_size` used in HistoryBuffer.
        """
        result = {}
        for k, (v, itr) in self._latest_scalars.items():
            result[k] = (
                self._history[k].median(self.count_samples(k, window_size))
                if self._smoothing_hints[k]
                else v,
                itr,
            )
        return result

    def count_samples(self, name, window_size=20):
        """
        Return the number of samples logged in the past `window_size` iterations.
        """
        samples = 0
        data = self._history[name].values()
        for _, iter_ in reversed(data):
            if iter_ > data[-1][1] - window_size:
                samples += 1
            else:
                break
        return samples

    def smoothing_hints(self):
        """
        Returns:
            dict[name -> bool]: the user-provided hint on whether the scalar
                is noisy and needs smoothing.
        """
        return self._smoothing_hints

    def step(self):
        """
        User should either: (1) Call this function to increment storage.iter when needed. Or
        (2) Set `storage.iter` to the correct iteration number before each iteration.

        The storage will then be able to associate the new data with an iteration number.
        """
        self._iter += 1

    @property
    def iter(self):
        """
        Returns:
            int: The current iteration number. When used together with a trainer,
                this is ensured to be the same as trainer.iter.
        """
        return self._iter

    @iter.setter
    def iter(self, val):
        self._iter = int(val)

    @property
    def iteration(self):
        # for backward compatibility
        return self._iter

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()

    @contextmanager
    def name_scope(self, name):
        """
        Yields:
            A context within which all the events added to this storage
            will be prefixed by the name scope.
        
            The @contextmanager decorator converts a generator function into a
            context manager. It handles the __enter__ and __exit__ protocol
            automatically.

            Basic Example:

            from contextlib import contextmanager

            @contextmanager
            def my_context():
                print("Entering context")
                yield "some value"  # This becomes the 'as' value
                print("Exiting context")

            # Usage:
            with my_context() as value:
                print(f"Inside context with: {value}")

            # Output:
            # Entering context
            # Inside context with: some value
            # Exiting context    
        """
        old_prefix = self._current_prefix
        self._current_prefix = name.rstrip("/") + "/"
        yield
        self._current_prefix = old_prefix

    def clear_images(self):
        """
        Delete all the stored images for visualization. This should be called
        after images are written to tensorboard.
        """
        self._vis_data = []

    def clear_histograms(self):
        """
        Delete all the stored histograms for visualization.
        This should be called after histograms are written to tensorboard.
        """
        self._histograms = []
