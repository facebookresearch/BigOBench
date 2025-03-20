# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

def read_jsonl_file(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    
    Args:
        file_path (str): The path to the JSONL file.
    
    Returns:
        list: A list of dictionaries, where each dictionary represents a JSON object.
    """
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def correct_complexity_formatting(complexity):
    if complexity[0] == "o" or complexity[0] == "O":
        complexity = complexity[1:]

    while len(complexity) >= 2 and complexity[0] == "(" and complexity[-1] == ")":
        complexity = complexity[1:-1]

    return f"O({complexity})"

def save_dict_to_json(data, path):
    """
    Saves the given dictionary to a JSON file at the specified path.
    If the file already exists, its contents are updated with the new data.
    Args:
        data (dict): The dictionary to be saved.
        path (str): The path where the JSON file will be saved.
    Returns:
        None
    """
    if os.path.exists(path):
        try:
            with open(path, 'r') as file:
                existing_data = json.load(file)
                existing_data.update(data)
                data = existing_data
        except json.JSONDecodeError:
            print(f"Warning: File {path} is not a valid JSON. Overwriting its contents.")
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)

def avg_dict(
    keys: List[str], dictionary: Dict[str, List[float]], dist: bool
) -> Dict[str, float]:
    avg = {}

    for k in keys:
        vs = dictionary[k]
        avg_v = float(np.mean(vs)) if len(vs) > 0 else 0.0
        no_vs = len(vs) == 0

        if dist:
            raise Exception('not handled')

        if no_vs:
            avg_v = -1

        avg[k] = avg_v

    return avg

def aggregate_at_k(
    metrics: Dict[str, List[Any]],
    at_metrics: List[str],
    n_samples: int,
    pass_ats: List[int],
    avg_metrics: Optional[List[str]] = None,
    inconsistent = False,
    metrics_for_list = ["time", "space"],
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

    # metrics_by_example: metric_name -> task_id -> list of results

    metrics_at_k = defaultdict(list)
    metrics_to_task_id_dict = defaultdict(list)

    for m, by_example in metrics_by_example.items():
        for task_id, results in by_example.items():
            assert inconsistent or (n_samples == len(
                results
            )), f"Inconsistent number of samples for example {task_id=}"

            if m in avg_metrics:
                metrics_at_k[f"avg_{m}"].append(np.mean(results))
                metrics_to_task_id_dict[f"avg_{m}"].append(task_id)
            else:
                metrics_at_k[f"any_{m}"].append(100 if sum(results) > 0 else 0)
                metrics_to_task_id_dict[f"any_{m}"].append(task_id)

                if m in metrics_for_list:
                    metrics_at_k[f"list_{m}"].append(results)
                    metrics_to_task_id_dict[f"list_{m}"].append(task_id)

                for k in pass_ats:
                    pk = pass_at_k(len(results), sum(results) / 100, k) * 100
                    metrics_at_k[f"{m}_at_{k}"].append(pk)
                    metrics_to_task_id_dict[f"{m}_at_{k}"].append(task_id)

    return metrics_at_k, metrics_to_task_id_dict

def aggregate_dict_list_into_list_dict(question_list, metric_name_list):
    metric_dict = {"raw": []}

    for attribute in metric_name_list:
        metric_dict[f"{attribute}_at_1"] = []

    assert len(question_list) > 0

    for question in question_list:
        
        for metric_name in metric_name_list:
            metric_dict[f"{metric_name}_at_1"].append(question[f"{metric_name}_at_1"])
            
        metric_dict["raw"].append(question["raw"])

    return metric_dict

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