# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, "../../")
sys.path.insert(0, src_dir)

import json

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import copy, deepcopy
from dataclasses import dataclass
from logging import WARNING, getLogger
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Optional
from unittest.mock import Mock, patch
import random

import httpx
from openai import OpenAI
from tenacity import retry
from tenacity.stop import stop_after_attempt
from tenacity.wait import wait_random_exponential, wait_fixed
from tenacity import Retrying, RetryError, stop_after_attempt

from src.eval.iterator.api import Packet
from src.eval.iterator.args import ValidArgs
from src.eval.iterator.runner import (
    _append_generation_results,
    multiturn_task_evaluation,
)
from src.eval.task import get_task
from src.eval.task.base import MultiturnTask
from src.eval.iterator.task_iterator import TaskIterator
from src.eval.iterator.logger import initialize_logger
from src.eval.iterator.params import Params, cfg_from_cli
from src.eval.iterator.utils import (
    avg_dict,
    gather_and_save_data,
    load_eval_data,
    log_host,
    reload_processed_examples,
    setup_env,
)

# from eval.task import TaskRegistry

logger = getLogger()
getLogger("filelock").setLevel(WARNING)
DEFAULT_MODEL = "meta-llama/Llama-3.1-70B-Instruct"

@dataclass
class TPEvalArgs(Params):
    task: ValidArgs
    dump_dir: str
    host: str = None
    progress: bool = False
    seed: int = 42
    max_concurrent_requests: int = 5
    log_level: str = "info"
    model: str = DEFAULT_MODEL
    max_tokens: int = 16384
    light_request_arguments: bool = False


@dataclass
class OpenAIGenArgs:
    model: str
    use_sampling: bool
    temperature: float
    host: str = None
    top_p: float = 1.0
    max_concurrent_requests: int = 1
    max_tokens: int = 16384
    light_request_arguments: bool = False


class OpenAIGen():
    def __init__(self, args: OpenAIGenArgs):
        super().__init__()
        if args.host is None or args.host == "":
            print("Using OpenAI API")

            self.client_list = [
                OpenAI(
                    api_key = os.environ.get("OPENAI_API_KEY"),
                    max_retries = 1,
                    timeout = httpx.Timeout(timeout=10.0, connect=5.0)
                )
            ]

        else:
            host_list = args.host.split(',')
            url_list = [f"http://{host}:8000/v1" for host in host_list]
            self.client_list = [
                OpenAI(
                    base_url=url, 
                    api_key="EMPTY",
                    max_retries = 1,
                    timeout = httpx.Timeout(timeout=5000.0, connect=5.0),
                ) 
                for url in url_list
            ]

            print(f"Running VLLM with {len(self.client_list)} instances")

        self.args = args

    def generate(self, q: Queue[Optional[Packet]]):
        futures: dict[Any, Any] = {}

        def yield_completed(block=False):
            nonlocal futures
            for thread_id in list(futures.keys()):
                future = futures[thread_id]
                if not block:
                    if not future.done():
                        continue
                ret = future.result()
                reply = ret.choices[0].message.content
                res = Packet(
                    thread_id=thread_id,
                    text=reply,
                )
                del futures[thread_id]
                yield res

        with ThreadPoolExecutor(
            max_workers=self.args.max_concurrent_requests
        ) as executor:
            while True:
                try:
                    packet = q.get(timeout=0.1)
                    if packet is None:
                        break
                except Empty:
                    for r in yield_completed():
                        yield r
                    continue

                dialog = packet.dialog 

                messages = [
                    {"role": message.source, "content": message.body}
                    for message in dialog
                ]

                # filter down messages role etc that should not be sent to vllm
                messages = list(
                    filter(
                        lambda x: x["role"] in ["user", "system", "assistant"], messages
                    )
                )
                # assert messages[0]['role'] == 'system'
                messages = [messages[0]] + list(
                    filter(lambda x: x["role"] in ["user", "assistant"], messages[1:])
                )
                if messages[-1]["role"] == "assistant":
                    messages.pop()

                assert not packet.thread_id in futures
                futures[packet.thread_id] = executor.submit(
                    self.call_chat,
                    self.args.model,
                    messages,
                    temperature=self.args.temperature if self.args.use_sampling else 0,
                    top_p=self.args.top_p,
                    max_tokens=self.args.max_tokens,
                    system_message=None,
                )
                for r in yield_completed():
                    yield r
            for r in yield_completed(block=True):
                yield r

    def call_chat(
        self,
        model_name_or_path,
        messages,
        temperature,
        top_p,
        max_tokens,
        system_message,
        **model_args,
    ):
        """
        Calls the OpenAI API to generate completions for the given inputs using the new API interface.
        Args:
            model_name_or_path (str): The name or path of the model to use.
            inputs (str): The inputs to generate completions for.
            temperature (float): The temperature to use.
            top_p (float): The top_p to use.
            **model_args (dict): Additional model arguments.
        Returns:
            tuple: A tuple containing the response and the cost of the completion.
        """

        try:
            for attempt_index, attempt in enumerate(Retrying(wait=wait_random_exponential(min=30, max=600), stop=stop_after_attempt(3))):
                # print('tryping an attempt', attempt_index)

                with attempt:
                    client = random.choice(self.client_list)
                    try:
                        response = (
                            client.chat.completions.create(
                                model=model_name_or_path,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                **model_args,
                            ) 
                            if not self.args.light_request_arguments 
                            else client.chat.completions.create(
                                model=model_name_or_path,
                                messages=list(
                                    filter(
                                        lambda x: x["role"] in ["user", "assistant"], messages
                                    )
                                ),
                                max_completion_tokens=max_tokens,
                            ) 
                        )

                        # cost = calc_cost(model_name_or_path, input_tokens, output_tokens)
                        return response  # , cost

                    except Exception as e:
                        print(f"{client.base_url} API Error {attempt_index}: {e}")
                        raise

        except RetryError:
            for attempt_index, attempt in enumerate(Retrying(wait=wait_fixed(1), stop=stop_after_attempt(6))):
                print('tryping an attempt with default behavior', attempt_index)
                with attempt:
                    client = random.choice(self.client_list)
                    messages = deepcopy(messages)

                    # In certain cases, reasoning does not converge so we conclude with a dummy request
                    for i, x in enumerate(messages):
                        if x["role"] == "user":
                            messages[i]["content"] = "Hi !"

                    try:
                        response = (
                            client.chat.completions.create(
                                model=model_name_or_path,
                                messages=messages,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                **model_args,
                            ) 
                            if not self.args.light_request_arguments 
                            else client.chat.completions.create(
                                model=model_name_or_path,
                                messages=list(
                                    filter(
                                        lambda x: x["role"] in ["user", "assistant"], messages
                                    )
                                ),
                                max_completion_tokens=max_tokens,
                            ) 
                        )

                        return response  # , cost
                        
                    except Exception as e:
                        print(f"{client.base_url} API Error {attempt_index} ON RETRY: {e}")
                        raise


def tp_task_evaluation(
    task_name: str,
    args: ValidArgs,
    dump_dir: str,
    seed: int,
    host: str,
    model: str,
    max_tokens: str,
    progress: bool = False,
    max_concurrent_requests: int = 5,
    light_request_arguments: bool = False,
) -> dict[str, float]:
    metrics_ls: dict[str, list[Any]] = defaultdict(list)
    task_name, task = get_task(args.data_file_path, task_name)
    eval_path = args.data_file_path
    dataset_name = f"{task_name}-{Path(task.data_file_path).stem}"

    if (full_results := load_eval_data(dataset_name, dump_dir)) is not None:
        logger.info(f"Eval for {task_name} already run")
        # full_results contains the eval results for the global run
        # so the average below does not need to be distributed
        _append_generation_results(metrics_ls, full_results)
        metrics_ls = task.aggregate_metrics(metrics_ls)
        metrics = avg_dict(task.metrics, metrics_ls, dist=False)
        return metrics
    
    # TODO: resuming!
    data_iterator = TaskIterator(
        path=eval_path,
        task=task,
        world_rank=0,
        world_size=1,
        seed=seed,
    )
    # find out if we are resuming a partial run
    ids, examples = reload_processed_examples(
        dataset_name=dataset_name,
        dump_dir=dump_dir,
        world_rank=0,
    )
    if (navail := len(examples)) > 0:
        logger.info(f"Found results for {navail} examples")
        _append_generation_results(metrics_ls, examples)
        data_iterator.skip(ids)
    items = list(data_iterator.example_iterator(ddp_layout="interleaved"))

    generator = OpenAIGen(
        args=OpenAIGenArgs(
            use_sampling=args.use_sampling,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            max_concurrent_requests=max_concurrent_requests,
            light_request_arguments=light_request_arguments,
            model=model,
            host=host,
        ),
    )
    if isinstance(task, MultiturnTask):
        multiturn_task_evaluation(
            g=generator,
            task_name=task_name,
            task=task,  # type: ignore
            args=args,
            dump_dir=dump_dir,
            world_rank=0,
            items=items,
            seed=seed,
            progress=progress,
            metrics_ls=metrics_ls,
            num_eval_threads=args.num_eval_threads,
        )
    else:
        raise RuntimeError(f"Unknown task type: {task}")
    # Gather data and aggregate to compute metrics
    logger.info(f"Saving data in {dump_dir}")
    gather_and_save_data(
        dataset_name=dataset_name,
        dump_dir=dump_dir,
        world_rank=0,
        world_size=1,
    )
    full_results = load_eval_data(dataset_name, dump_dir)
    assert full_results is not None
    metrics_ls.clear()
    _append_generation_results(metrics_ls, full_results)
    metrics_ls = task.aggregate_metrics(metrics_ls)
    metrics = avg_dict(task.metrics, metrics_ls, dist=False)
    return metrics

def get_absolute_path(path):
    """
    Returns the absolute path of the given path.
    If the path is already absolute, it is returned as is.
    Otherwise, it is transformed to an absolute path relative to the current file path.
    Args:
        path (str): The path to be transformed.
    Returns:
        str: The absolute path.
    """
    if os.path.isabs(path):
        return path
    else:
        current_file_dir = Path(__file__).parent.resolve()
        return str(current_file_dir / path)


def main(eval_args: TPEvalArgs):
    initialize_logger(eval_args.log_level)
    setup_env()
    log_host()

    args = eval_args.task
    # args.data_file_path = get_absolute_path(args.data_file_path)
    scores = {}
    os.makedirs(eval_args.dump_dir, exist_ok=True)
    with (Path(eval_args.dump_dir) / "eval_params.json").open("w") as fp:
        fp.write(eval_args.to_json())
    with patch("iterator.task_iterator.eval_rng", return_value=eval_args.seed), patch(
        "iterator.runner.eval_rng", return_value=eval_args.seed
    ), patch(
        "torch.distributed", Mock()
    ):
        for task in args.task_list:
            logger.info(f"Evaluating on {task} ...")
            metrics = tp_task_evaluation(
                task_name=task,
                args=args,
                dump_dir=eval_args.dump_dir,
                seed=eval_args.seed,
                progress=eval_args.progress,
                host=eval_args.host,
                model=eval_args.model,
                max_tokens=eval_args.max_tokens,
                light_request_arguments=eval_args.light_request_arguments,
                max_concurrent_requests=eval_args.max_concurrent_requests,
            )
            log = " - ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])
            logger.info(f"Results on {task}: {log}")
            for k, v in metrics.items():
                scores[f"task/{task}/{k}"] = v

    res_path = os.path.join(eval_args.dump_dir, "temp_results.json")
    with open(res_path, "w") as fp:
        logger.info(f"Dumping results in {res_path}")
        json.dump(scores, fp, sort_keys=True, indent=4)
    log = " - ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
    logger.info(f"ALL RESULTS: {log}")


if __name__ == "__main__":
    cfg: TPEvalArgs = cfg_from_cli(schema=TPEvalArgs)
    main(cfg)
