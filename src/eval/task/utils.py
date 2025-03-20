# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import ast
import logging
import os
import re
import string
from collections import Counter, defaultdict
from copy import copy
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from numpy.random import RandomState
from tree_sitter import Language, Node, Parser

from src.eval.iterator.dialog import Message

logger = logging.getLogger()

SYSTEM_PROMPT_CODE_ONLY = Message(
    source="system", body="Environment: ipython\nTools: none"
)

# Default pass_ats and other at metrics
DEFAULT_PASS_ATS = {"at_1": [1], "at_2": [1, 2], "at_5": [1, 2, 5], "at_10": [1, 2, 5, 10], "at_100": [1, 10, 20, 50, 100]}


def get_n_samples_for_pass_ats(n_samples: Optional[int], pass_ats: list[int]) -> int:
    """Automatically derive n_samples for the given pass_ats."""
    if n_samples is not None:
        return n_samples
    """Return n_samples aligned with the desired pass_ats."""
    if len(pass_ats) == 0:
        return 1
    elif len(pass_ats) == 1 and pass_ats[0] == 1:
        return 1
    else:
        return 2 * max(pass_ats)


def em(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]) -> float:
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]) -> float:
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(
    prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]
) -> float:
    return max(f1(prediction, gt, normalize_fn) for gt in ground_truths)


def exact_match_score(
    prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]
) -> float:
    return max(em(prediction, gt, normalize_fn) for gt in ground_truths)


def edit_similarity_score(
    prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]
) -> float:
    return max(
        SequenceMatcher(None, normalize_fn(prediction), normalize_fn(gt)).ratio()
        for gt in ground_truths
    )

def different_seed_when_job_array(seed: int) -> int:
    # when using array jobs, we want different seed for each job
    return seed + int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


def check_python_syntax_correctness(code: str) -> bool:
    """simple python code syntax check"""
    try:
        # Warning from Python docs:
        # > It is possible to crash the Python interpreter with a sufficiently
        # > large/complex string due to stack depth limitations in Python's AST
        # > compiler.
        ast.parse(str(code))
    except BaseException:
        return False
    return True



def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased pass@k estimator from Codex (https://arxiv.org/abs/2107.03374).
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def pass_n_at_k_filtered(
    total: int,
    f: int,
    c: int,
    n: int,
    k: int,
    n_boot: int = 10000,
    rng: RandomState | None = None,
) -> float:
    """
    n@k solve/pass rate estimator based on filtering n of k samples. Assumes
    correct samples are a subset of the filtered samples.
    See AlphaCode, Algorithm 1 (https://arxiv.org/abs/2203.07814).
    :param total: total number of samples
    :param f: number of filtered samples (e.g., according to public tests)
    :param c: number of correct samples
    :param n: number of allowed submissions, n in n@k
    :param k: number of allowed samples, k in n@k
    :param n_boot: number of bootstraps for estimation
    :param rng: optional random state, initialize a new one if None
    """
    assert f <= total
    assert c <= total
    assert k <= total
    assert c <= f

    if rng is None:
        rng = np.random.RandomState()

    filtered = rng.hypergeometric(f, total - f, k, n_boot)
    n_p = np.minimum(filtered, n)
    # Avoid errors for filtered==0, we can count those as sure failures
    selected = rng.hypergeometric(c, f - c, n_p[np.where(n_p > 0)])
    failures = np.where(n_p == 0)[0].shape
    return np.mean(np.concatenate([selected > 0, np.zeros(failures)]))


def batch_duplicate(examples: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """
    Duplicate examples for eventual aggregation with aggregate_at_k().
    """
    duplicates: List[Dict[str, Any]] = []
    for example in examples:
        for i in range(n):
            duplicates.append(copy(example))
            duplicates[-1]["sample"] = i
    return duplicates


def metric_names_at_k(
    at_metrics: List[str],
    n_samples: int,
    pass_ats: List[int],
    avg_metrics: Optional[List[str]] = None,
) -> List[str]:
    """
    List metrics as computed by aggregate_at_k.
    """
    ret: List[str] = []
    for k in pass_ats:
        ret += [f"{metric}_at_{k}" for metric in at_metrics]
    if n_samples > 1:
        ret += [f"any_{metric}" for metric in at_metrics]
    if avg_metrics:
        if n_samples == 1:
            ret += [metric for metric in avg_metrics]
        else:
            ret += [f"avg_{metric}" for metric in avg_metrics]
    return ret


def aggregate_at_k(
    metrics: Dict[str, List[Any]],
    at_metrics: List[str],
    n_samples: int,
    pass_ats: List[int],
    avg_metrics: Optional[List[str]] = None,
) -> Dict[str, List[Any]]:
    """
    Aggregate *_at_1 metrics into *_at_k via pass_at_k().
    """
    if avg_metrics is None:
        avg_metrics = []

    metrics_by_example: Dict[str, Dict[str, List[Any]]] = {
        key: defaultdict(list) for key in (at_metrics + avg_metrics)
    }
    for i, raw in enumerate(metrics["raw"]):
        for m in at_metrics:
            metrics_by_example[m][raw["task_id"]].append(metrics[f"{m}_at_1"][i])
        for m in avg_metrics:
            metrics_by_example[m][raw["task_id"]].append(metrics[m][i])

    metrics_at_k = defaultdict(list)
    for m, by_example in metrics_by_example.items():
        for task_id, results in by_example.items():
            assert n_samples == len(
                results
            ), f"Inconsistent number of samples for example {task_id=}"
            if m in avg_metrics:
                metrics_at_k[f"avg_{m}"].append(np.mean(results))
            else:
                metrics_at_k[f"any_{m}"].append(100 if sum(results) > 0 else 0)
                for k in pass_ats:
                    pk = pass_at_k(len(results), sum(results) / 100, k) * 100
                    metrics_at_k[f"{m}_at_{k}"].append(pk)

    return metrics_at_k