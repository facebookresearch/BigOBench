# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import copy
from hashlib import md5
from typing import Any, Dict, Iterator
import numpy as np
from functools import partial

from src.eval.task.base import BaseTask
from src.eval.iterator.utils import eval_rng
from src.eval.iterator.jsonl import JSONLIterator


def make_example(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures that the data has a task_id entry"""
    if "task_id" in data:
        return data
    elif "metadata" in data and "task_id" in data["metadata"]:
        data["task_id"] = data["metadata"]["task_id"]
        return data

    try:
        # For chat formats, use sample_id in metadata
        task_id = data["metadata"]["sample_id"]
        cdata = copy(data)
        cdata["task_id"] = task_id
        return cdata
    except:
        pass

    # task_id as hash of example contents; https://stackoverflow.com/a/42151923
    def make_hashable(o):
        if isinstance(o, (tuple, list)):
            return tuple((make_hashable(e) for e in o))
        if isinstance(o, dict):
            return tuple(sorted((k, make_hashable(v)) for k, v in o.items()))
        if isinstance(o, (set, frozenset)):
            return tuple(sorted(make_hashable(e) for e in o))
        return o

    hasher = md5()
    hasher.update(repr(make_hashable(data)).encode())
    cdata = copy(data)
    cdata["task_id"] = hasher.hexdigest()
    return cdata


class TaskIterator:
    def __init__(
        self,
        path: str,
        task: BaseTask,
        world_rank: int,
        world_size: int,
        seed: int,
    ):
        self.world_rank = world_rank
        self.world_size = world_size
        self.path = path
        self.task = task
        self.rng = eval_rng(
            base_seed=seed,
            different_seed_for_mp_groups=True,
            different_seed_for_tasks_in_job_array=True,
        )
        self.skip_set = set[tuple[str, int]]()

    def skip(self, ids: set[tuple[str, int]]) -> None:
        self.skip_set = ids

    def example_iterator(self, ddp_layout: str) -> Iterator[Dict]:
        # Load all examples, then do batch_preprocess(), and only then assign to
        # all ranks for DDP. This way we even out the load for pass_at_k which
        # duplicates in batch_preprocess().
        examples = [
            make_example(ex)
            for ex in JSONLIterator(
                fpath=self.path,
                world_rank=0,
                world_size=1,
                infinite=False,
            )
        ]
        example_ids = set()
        for example in examples:
            assert (
                example["task_id"] not in example_ids
            ), f"Duplicate task_id: {example['task_id'], example}"
            example_ids.add(example["task_id"])

        transformed_examples = self.task.batch_preprocess(examples)

        if ddp_layout == "interleaved":
            local_examples = transformed_examples[self.world_rank :: self.world_size]

        elif ddp_layout == "chunked":
            N = len(transformed_examples)
            n = int(np.ceil(N / self.world_size))
            rank = self.world_rank
            mod = N % self.world_size
            if rank < mod:
                local_examples = transformed_examples[rank * n : (rank + 1) * n]
            else:
                n2 = int(np.floor(N / self.world_size))
                rank2 = rank - mod
                local_examples = transformed_examples[mod * n :][
                    rank2 * n2 : (rank2 + 1) * n2
                ]
        else:
            raise ValueError(f"Unknown {ddp_layout=}")

        filtered_examples = filter(
            lambda e: (e["task_id"], e.get("sample", 0)) not in self.skip_set,
            local_examples,
        )
        process_rng_fn = partial(self.task.process, rng=self.rng)
        return map(process_rng_fn, filtered_examples)