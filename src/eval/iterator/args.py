# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import List
from src.eval.iterator.params import Params
from src.eval.task import TaskRegistry

logger = getLogger()

@dataclass
class ValidArgs(Params):
    # tasks to evaluate
    ppl_files_str: str = ""  # comma separated list of files to eval PPL
    tasks_root_dir: str = ""  # tasks root directory
    data_file_path : str = ""
    tasks_str: str = ""  # comma separated list of tasks

    write_eval: bool = False
    batch_size: int = 32
    seq_len: int = 2048
    n_batches: int = 100  # number of batches used for PPL evaluation
    num_eval_threads: int = 5  # parallel metrics computation (if supported)

    # decoding parameters
    use_sampling: bool = False
    temperature: float = 1.0
    temperature_min: float = -1.0  # uniformly sample temperature if these two are >=0
    temperature_max: float = -1.0
    n_samples: int = 1
    top_k: int = 0
    top_p: float = 0.0
    user_eos: str = ""
    token_healing: bool = False
    generation: str = "vllm"

    # prompts for generation tests
    prompt_path: str = ""  # prompt path
    max_prompt_len: int = 256  # maximum length of prompt in run_generations

    # for MATH - TODO clean that with params per task maybe ?
    majority_voting: int = 0
    random_fewshots: bool = False

    # minimize data for evaluation tasks, useful to set to false for debugging
    minimize_data: bool = True

    debug: bool = False

    @property
    def ppl_files(self) -> List[str]:
        paths = [path for path in self.ppl_files_str.split(",") if len(path) > 0]
        return [path for path in paths]

    @property
    def task_list(self) -> List[str]:
        tasks = [task for task in self.tasks_str.split(",") if len(task) > 0]
        assert len(tasks) == len(set(tasks))

        available_tasks = set(TaskRegistry.names())
        unavailable_tasks = set(tasks) - available_tasks
        if unavailable_tasks:
            raise ValueError(
                f"Could not import tasks {unavailable_tasks}.\n"
                f"The available tasks are {available_tasks}. "
                "Check the rest of the logs for more information and check that these "
                "tasks were registered through @register_task in src/eval/task"
            )
        return tasks

    def should_do_eval(self):
        return len(self.ppl_files) > 0 or len(self.task_list) > 0

    def __post_init__(self):
        # decoding params
        assert self.temperature >= 0
        assert self.top_k >= 0
        assert 0 <= self.top_p <= 1

        # ppl files
        if len(self.ppl_files) > 0:
            assert len(self.ppl_files) == len(set(self.ppl_files))
            assert all(path.endswith(".jsonl") for path in self.ppl_files)
            assert all(Path(path).is_file() for path in self.ppl_files), [
                path for path in self.ppl_files if not Path(path).is_file()
            ]

        # tasks root dir
        if len(self.tasks_root_dir) > 0:
            self.tasks_root_dir = self.tasks_root_dir
            assert os.path.isdir(self.tasks_root_dir), self.tasks_root_dir

        # generation task
        if self.prompt_path:
            assert self.prompt_path.endswith(".jsonl"), self.prompt_path
            assert os.path.isfile(self.prompt_path), self.prompt_path
