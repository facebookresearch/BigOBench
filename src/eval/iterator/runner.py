# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from copy import copy, deepcopy
from dataclasses import asdict
from logging import getLogger
from typing import Any, Dict, List
from pathlib import Path

from src.eval.iterator.api import NoErr, Packet
from src.eval.iterator.args import ValidArgs
from src.eval.task.base import MultiturnTask
from src.eval.iterator.utils import (
    minimize_eval_datum,
    save_cur_data
)
from src.eval.iterator.utils import eval_rng

logger = getLogger()

def _append_generation_results(
    metrics_ls: Dict[str, List[Any]], results: List[Any]
) -> None:
    for res in results:
        metrics_ls["raw"].append(res["raw"])
        for key, value in res.items():
            if key in ["raw", "dialog"]:
                continue
            metrics_ls[key].append(value)


def multiturn_task_evaluation(
    g,
    task_name: str,
    task: MultiturnTask,
    args: ValidArgs,
    dump_dir: str,
    world_rank: int,
    items: List[Dict],
    seed: int,
    progress: bool,
    metrics_ls: Dict[str, list[Any]],
    num_eval_threads: int = 0,
) -> None:
    rng = eval_rng(
        base_seed=seed,
        different_seed_for_mp_groups=True,
        different_seed_for_tasks_in_job_array=True,
    )
    q = queue.Queue[Packet | None]()

    orig_items = copy(items[::-1])
    items = []
    n_queued = 0

    def add_to_queue(idx=None):
        nonlocal n_queued

        if idx is None:
            q.put(None)
            return

        item = items[idx]
        dialog = task.get_prompt_tokens(item["dialog"])
        
        if args.temperature_min >= 0 and args.temperature_max >= 0:
            temperature = rng.uniform(
                low=args.temperature_min, high=args.temperature_max
            )
            item.setdefault("temperatures", []).append(temperature)
        else:
            temperature = None

        pkt = Packet(idx, dialog=dialog, temperature=temperature)
        q.put(pkt)

        n_queued += 1

    for _ in range(args.batch_size):
        if not orig_items:
            break
        items.append(orig_items.pop())
        add_to_queue(len(items) - 1)
    if not items:
        # No work to do
        return

    results_lock = threading.Lock()

    def eval_dialog(dialog, item):
        sample_metrics = task.evaluate(dialog, example=item["raw"])

        item.update(sample_metrics)
        ser_item = copy(item)
        del ser_item["dialog"]
        ser_item["dialog"] = [asdict(m) for m in dialog.messages]
        min_item = minimize_eval_datum(ser_item, list(sample_metrics.keys())) if args.minimize_data else ser_item

        with results_lock:
            save_cur_data(
                data=[min_item],
                dataset_name=f"{task_name}-{Path(task.data_file_path).stem}",
                dump_dir=dump_dir,
                world_rank=world_rank,
            )

            _append_generation_results(metrics_ls, [item])

    start = time.time()

    with (
        ThreadPoolExecutor(max_workers=num_eval_threads)
        if task.has_threadsafe_evaluate and num_eval_threads > 0
        else nullcontext()  # type: ignore[attr-defined]
    ) as executor:
        for cnt, pkt in enumerate(g.generate(q)):
            n_queued -= 1

            item = items[pkt.thread_id]
            dialog = item["dialog"]

            task.add_response(dialog, pkt.text, example=item["raw"])
            if not dialog.done:
                # Generate another response
                add_to_queue(pkt.thread_id)
                continue

            if executor is not None:
                executor.submit(eval_dialog, dialog, item)
            else:
                eval_dialog(dialog, item)

            # Push next item to queue, or None to signal end of loop if we've
            # fully processed all items.
            if orig_items:
                items.append(orig_items.pop())
                add_to_queue(len(items) - 1)
            elif n_queued == 0:
                add_to_queue(None)
