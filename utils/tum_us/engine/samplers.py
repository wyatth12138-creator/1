"""Homogeneous batches with equal probability for each task, then each dataset."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterator

import torch
from torch.utils.data import Sampler


def dataset_groups(dataset) -> dict[tuple[str, str], list[int]]:
    if getattr(dataset, "groups", None):
        return {tuple(key): list(indices) for key, indices in dataset.groups.items() if indices}
    groups = defaultdict(list)
    for index, record in enumerate(dataset.records):
        groups[(record["task"], record["dataset_id"])].append(index)
    return dict(groups)


class BalancedGroupBatchSampler(Sampler[list[int]]):
    """Sample tasks and datasets uniformly; shuffle/recycle samples inside a group.

    Sampling uses its own epoch-derived generator, so restoring an epoch and skipping
    completed batches is exact and does not consume model/dropout RNG state.
    """
    def __init__(self, dataset, batch_size: int = 1, steps_per_epoch: int | None = None,
                 seed: int = 3407):
        self.groups = dataset_groups(dataset)
        if not self.groups:
            raise ValueError("No records match the dataset filters.")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.steps_per_epoch = int(steps_per_epoch or math.ceil(len(dataset) / self.batch_size))
        self.seed, self.epoch = int(seed), 0
        self.tasks = sorted({task for task, _ in self.groups})
        self.by_task = {task: sorted(key for key in self.groups if key[0] == task) for task in self.tasks}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003)
        queues: dict[tuple[str, str], list[int]] = {}
        task_queue: list[str] = []
        dataset_queues: dict[str, list[tuple[str, str]]] = {}
        for _ in range(self.steps_per_epoch):
            if not task_queue:
                task_queue = [self.tasks[i] for i in torch.randperm(len(self.tasks), generator=generator).tolist()]
            task = task_queue.pop()
            keys = self.by_task[task]
            if not dataset_queues.get(task):
                dataset_queues[task] = [keys[i] for i in torch.randperm(len(keys), generator=generator).tolist()]
            key = dataset_queues[task].pop()
            batch = []
            while len(batch) < self.batch_size:
                if not queues.get(key):
                    source = self.groups[key]
                    order = torch.randperm(len(source), generator=generator).tolist()
                    queues[key] = [source[i] for i in order]
                batch.append(queues[key].pop())
            yield batch


class SequentialGroupBatchSampler(Sampler[list[int]]):
    """Visit every evaluation record exactly once, without cross-dataset padding."""
    def __init__(self, dataset, batch_size: int = 1):
        self.groups = dataset_groups(dataset)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    def __len__(self) -> int:
        return sum(math.ceil(len(indices) / self.batch_size) for indices in self.groups.values())

    def __iter__(self) -> Iterator[list[int]]:
        for key in sorted(self.groups):
            indices = self.groups[key]
            for start in range(0, len(indices), self.batch_size):
                yield indices[start:start + self.batch_size]
