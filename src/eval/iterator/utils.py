# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
from typing import Union, Tuple, List, Dict, Any, Optional

from pathlib import Path
import os
import glob
import json
import torch
import atexit
import shutil
import socket
import logging
import tempfile
import numpy as np
import torch.distributed as dist
from numpy.random import RandomState


import sys
from pathlib import Path

# code work as expected (from src.foo import bar)
sys.path.insert(0, str(Path(__file__).parent.parent))

from complexity.sandbox.server import ForkServer

Scalar = Union[int, float]

logger = logging.getLogger()

def avg_dict(
    keys: List[str], dictionary: Dict[str, List[float]], dist: bool
) -> Dict[str, float]:
    avg = {}

    for k in keys:
        vs = dictionary[k]
        avg_v = float(np.mean(vs)) if len(vs) > 0 else 0.0
        no_vs = len(vs) == 0

        if dist:
            raise Exception()

        if no_vs:
            logger.warning(f"Error when computing metric {k}: metric never appears")
            avg_v = -1

        avg[k] = avg_v

    return avg


def minimize_eval_datum(d: Dict[str, Any], optional_keys = []) -> Dict[str, Any]:
    """Strip example for serialization."""
    include = {
        "prompt",
        "generation",
        "results",
        "temperature",
        "temperatures",
        "dialog",
    }
    include = include | set(optional_keys)
    min_d = {k: v for k, v in d.items() if k in include}
    min_d["raw"] = {"task_id": d["raw"]["task_id"]}
    if "sample" in d["raw"]:
        min_d["raw"]["sample"] = d["raw"]["sample"]
    return min_d


def reload_processed_examples(
    dataset_name: str,
    dump_dir: str,
    world_rank: int,  # data parallel rank
) -> Tuple[set[Tuple[str, int]], List]:
    """
    If partial eval results are present, return them with a set
    of (id, sample) pairs for the results available.
    """
    save_path = Path(dump_dir) / "eval_results" / dataset_name / f"{world_rank}.jsonl"

    data = []
    ids = set()

    if save_path.exists():
        with open(save_path, "r") as f:
            for l in f:
                obj = json.loads(l)
                raw = obj["raw"]
                oid = (raw["task_id"], raw.get("sample", 0))
                if oid in ids:
                    logger.warning(
                        f"Example id {oid} appears twice in results, skipping them"
                    )
                    save_path.unlink(missing_ok=True)
                    return set(), []
                ids.add(oid)
                data.append(obj)

    return ids, data


def load_eval_data(dataset_name: str, dump_dir: str) -> Optional[List]:
    res_path = Path(dump_dir) / "eval_results" / f"{dataset_name}.jsonl"
    if not res_path.exists():
        return None

    data = []
    with open(res_path, "r") as f:
        for l in f:
            data.append(json.loads(l))

    return data


def save_cur_data(
    data: List,
    dataset_name: str,
    dump_dir: str,
    world_rank: int,  # get_data_parallel_rank()
) -> None:
    assert isinstance(data, list), data

    cur_save_dir = Path(dump_dir) / "eval_results" / dataset_name
    cur_save_dir.mkdir(parents=True, exist_ok=True)
    cur_save_path = cur_save_dir / f"{world_rank}.jsonl"
    with open(cur_save_path, "a") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")


def gather_and_save_data(
    dataset_name: str,
    dump_dir: str,
    world_rank: int,  # get_data_parallel_rank()
    world_size: int,  # get_data_parallel_world_size()
) -> None:
    assert 0 <= world_rank < world_size
    torch.distributed.barrier()

    if world_rank != 0:
        return
    save_dir = Path(dump_dir) / "eval_results"
    tmp_save_path = save_dir / f"{dataset_name}_tmp.jsonl"
    save_path = save_dir / f"{dataset_name}.jsonl"
    cur_save_dir = save_dir / dataset_name
    results_path = sorted(glob.glob(str(cur_save_dir / "*.jsonl")))
    with open(tmp_save_path, "w") as f:
        for path in results_path:
            with open(path, "r") as g:
                for line in g:
                    # check that each line is a json.dumps
                    f.write(json.dumps(json.loads(line)))
                    f.write("\n")
    tmp_save_path.rename(save_path)
    logger.info(f"Data saved at {save_path}")
    shutil.rmtree(str(cur_save_dir))


def setup_env():
    assert (
        os.environ.get("OMP_NUM_THREADS") == "1"
    ), "expected to run with OMP_NUM_THREADS=1"

    if "USE_RESILIENT" in os.environ:
        raise NotImplementedError("Resilient support has been disabled")

    triton_cache_dir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, triton_cache_dir, ignore_errors=True)
    logger.debug(f"Setting TRITON_CACHE_DIR to {triton_cache_dir}")
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir

    # Initialize code evaluation server
    _ = ForkServer.global_instance()





def flatten_dict(d: Dict, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




def log_host():
    # logging on stdout / stderr, helpful when debbuging some jobs
    logger.warning(f"Host: {socket.gethostname()}")
    logger.warning(f"Job hosts: {os.environ.get('SLURM_JOB_NODELIST', '')}")
    logger.warning(f"Slurm job id: {int(os.environ.get('SLURM_JOB_ID', -1))}")



def eval_rng(
    base_seed: int = 42,
    different_seed_for_mp_groups: bool = False,
    different_seed_for_tasks_in_job_array: bool = False,
) -> RandomState:
    """
    Please use this RandomState for creating the data in your eval tasks
    Base seed needs to be the same within a model parallel group

    different_seed_for_mp_groups: bool -> If you want different GPUs to
        have different seed, please set it to True (it will make sure that it is
        the same within the same mp group, but different between mp groups)
    different_seed_for_tasks_in_job_array -> If you want the seed to be different
       between tasks in job array (to have different prompts sampled for majority
       voting for instance)

    """
    seed = [base_seed]

    logger.info(f"Creating rng with seed: {tuple(seed)}")

    return RandomState(tuple(seed))
