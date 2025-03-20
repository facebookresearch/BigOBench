# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import importlib
from itertools import product
import logging
from pathlib import Path
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "../../../")
sys.path.insert(0, src_dir)

from src.eval.task.base import BaseTask
from src.eval.task.utils import DEFAULT_PASS_ATS


logger = logging.getLogger()


class AlreadyRegisteredTaskException(Exception):
    pass


class TaskRegistry:
    _REGISTRY: Dict[str, Callable[..., BaseTask]] = {}

    @staticmethod
    def names() -> AbstractSet[str]:
        return TaskRegistry._REGISTRY.keys()

    @staticmethod
    def register(name: str, callable: Callable[..., BaseTask]) -> None:
        if name in TaskRegistry._REGISTRY:
            raise AlreadyRegisteredTaskException(
                f"Already a task registered as {name}: {TaskRegistry._REGISTRY[name]} {callable}."
            )
        TaskRegistry._REGISTRY[name] = callable

    @staticmethod
    def build(name: str, **kwargs: Any) -> BaseTask:
        if name not in TaskRegistry._REGISTRY:
            raise ValueError(f"No task registered under the name {name}")
        return TaskRegistry._REGISTRY[name](**kwargs)

    @staticmethod
    def reset() -> None:
        TaskRegistry._REGISTRY = {}


def register_task(
    name: str, **parameters
) -> Callable[[Callable[..., BaseTask]], Callable[..., BaseTask]]:
    """Register the task name with the decorated task configuration callable."""

    def register(callable: Callable[..., BaseTask]) -> Callable[..., BaseTask]:
        if parameters is None:
            TaskRegistry.register(name, callable)
        else:
            task_name = name.format(**parameters)
            TaskRegistry.register(task_name, partial(callable, **parameters))
        return callable

    return register


def register_task_pass_at_k(
    name: str,
    pass_at_k: Optional[Dict[str, List[int]]] = None,
    n_samples: Optional[Dict[str, int]] = None,
    default_at_k: Optional[str] = None,
    **parameters,
) -> Callable[[Callable[..., BaseTask]], Callable[..., BaseTask]]:
    """Register the task name with the decorated task configuration callable for different pass at k.

    A base variant of the task will be registered with pass_ats value for 'default_at_k',
    and a new task is register for each of the specified 'pass_ats' with the pass_at key
    added as suffix to the task name and the corresponding pass_ats values.

    Note that this assumes that the task supports the "pass_ats" and "n_samples" arguments in its constructor.

    Example:
        @register_task_pass_at_k("human_eval", pass_ats={"at_1": [1], "at_10", [1, 10]}, default_at_k="at_1")
        will register the following tasks:
        * a base task "human_eval" with pass_ats=[1],
        * pass_at_k variant "human_eval:at_1" with pass_ats=[1],
        * pass_at_k variant "human_eval:at_10" with pass_ats=[1, 10],

    Args:
        name (str): Base name of the task. The pass_at_k variant will be suffixed with name:<at_k>
        pass_at_k (dict, optional): Dictionary of at_k key strings and list of pass_at_k values.
        n_samples (dict, optional): Dictionary of at_k key strings (expected to match pass_at_k keys)
            and the corresponding number of samples to use. Inferred automatically by the task if not specified.
        default_at_k (str, optional): at_k key to use as default for the base task. Default: "at_1".
        parameters: The rest of the parameters required to register the task.
    """

    # Use default at_k
    if pass_at_k is None:
        pass_at_k = DEFAULT_PASS_ATS

    if default_at_k is None:
        default_at_k = "at_1"

    assert (
        default_at_k is not None
        and pass_at_k is not None
        and (default_at_k in pass_at_k)
    ), f"Default pass_at={default_at_k} missing from {pass_at_k}"

    if n_samples is not None:
        assert pass_at_k is not None
        assert (
            n_samples.keys() == pass_at_k.keys()
        ), f"Mismatch between n_samples keys {n_samples.keys()} and pass_at_k keys {pass_at_k.keys()}"

    def register(callable: Callable[..., BaseTask]) -> Callable[..., BaseTask]:
        # register default task with pass_at_1 by default
        base_param_dict: Dict[str, Any] = {}
        base_param_dict.update(**parameters)
        base_param_dict["pass_ats"] = pass_at_k[default_at_k]
        base_task_name = name.format(**parameters) if parameters is not None else name
        TaskRegistry.register(base_task_name, partial(callable, **base_param_dict))
        # register tasks with pass_at_k variants as "task_name:at_k"
        for at_k, pass_ats in pass_at_k.items():
            name_at_k = f"{name}:{at_k}"
            param_dict: Dict[str, Any] = {}
            param_dict.update(**parameters)
            param_dict["pass_ats"] = pass_ats
            if n_samples is not None and at_k in n_samples:
                param_dict["n_samples"] = n_samples[at_k]
            task_name = name_at_k.format(**parameters)
            TaskRegistry.register(task_name, partial(callable, **param_dict))
        return callable

    return register


def get_task(data_file_path: str, task_name: str,):
    if task_name not in TaskRegistry.names():
        raise ValueError(f"Unknown task: {task_name}")
    task = TaskRegistry.build(
        task_name, data_file_path=data_file_path
    )
    return task_name, task


base_dir = Path(__file__).parent
for file in base_dir.rglob("*.py"):
    if not file.name.startswith("_"):
        relative_path = file.relative_to(base_dir).with_suffix("")
        module_path = ".".join(relative_path.parts)

        importlib.import_module(f"src.eval.task.{module_path}")
