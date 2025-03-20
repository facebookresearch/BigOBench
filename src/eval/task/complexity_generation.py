# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import sys
from collections import defaultdict
from functools import partial
from itertools import product
from typing import Any, List, Optional

import numpy as np

from src.eval.iterator.dialog import MessageV2
from src.complexity.execution_measures import ExecStatus
from src.complexity.execution_measures.worker_launcher import exec_python_iopairs
from src.eval.task import register_task_pass_at_k, TaskRegistry
from src.eval.task.base import (
    ChatMultiturnTask,
    Dialog,
    Example,
    ProcessedExample,
)
from src.eval.task.utils import (
    DEFAULT_PASS_ATS,
    aggregate_at_k,
    batch_duplicate,
    get_n_samples_for_pass_ats,
    metric_names_at_k,
    pass_n_at_k_filtered,
)

logger = logging.getLogger()

PROMPT_VARIANTS = [
    "time",
    "space",
    "time_ranking",
    "space_ranking",
    "no_conditioning",
    # "code_only_time_and_space_conditioned",
]


class ComplexityGenerationTask(ChatMultiturnTask):
    """
    Complexity Generation Task
    """
    has_threadsafe_evaluate: bool = True

    single_shot: bool = False
    feedback_format: str | None = None

    max_attempts = 1
    n_samples: int
    pass_ats: list[int]
    pass_n_at_ks: list[tuple[int, int]] = []
    at_metrics = [
        "pass",
        "public_pass",
        "compiles",
    ]

    # Formatting templates
    sys_prompt = "Environment: ipython\nTools: none"
    format_guide = "Your code should be enclosed in triple backticks like so: ```python YOUR CODE HERE ```. Use the backticks for your code only."

    n_fewshot_local: int = 0

    load_only_first_n: int = None

    def __init__(
        self,
        data_file_path: str,
        pass_ats: List[int],
        max_attempts: Optional[int] = None,
        single_shot: bool = False,
        feedback_format: Optional[str] = None,
        n_samples: Optional[int] = None,
        use_fewshot_local: bool = False,
        fewshot_file: str = None,
    ):
        super().__init__(data_file_path, MessageV2)

        self.fewshot_file = fewshot_file

        self._all_fewshot_examples = []

        if use_fewshot_local:
            self.n_fewshot_local = 3
        else:
            self.n_fewshot_local = 0

        # that's the number of samples you want to get per model query (for k > 2, we recommend getting 2 * k samples)
        self.n_samples = get_n_samples_for_pass_ats(n_samples, pass_ats)
        self.pass_ats = pass_ats

        # number of attempts you want in the multiturn setting !
        # if max_attempts is not None:
        #     self.max_attempts = max_attempts
        
        assert self.n_samples >= max(
            self.pass_ats
        ), f"Using pass_ats={self.pass_ats} but n_samples is too low: {self.n_samples}"

        # special setting to compare multiturn with single turn but as much compute, so do not use for simple inference
        # if single_shot:
        #     self.pass_n_at_ks.append((1, self.max_attempts))

        if self.n_samples > 1:
            for k in self.pass_ats:
                if k > 10:
                    self.pass_n_at_ks.append((10, k))
                elif k > 5:
                    self.pass_n_at_ks.append((5, k))

        if feedback_format is not None:
            assert feedback_format in PROMPT_VARIANTS
            self.feedback_format = feedback_format

    @property
    def get_attribute_fewshot_examples(
        self,
    ) -> List[Example]:
        if self.n_fewshot_local == 0:
            return []

        if len(self._all_fewshot_examples) == 0 and self.n_fewshot_local > 0:
            self._all_fewshot_examples = self._load_fewshot_file()[:self.n_fewshot_local]

        fewshot_examples = self._all_fewshot_examples

        return fewshot_examples

    # This gets called from TaskIterator, which is used in runner.py task_evaluation()
    def batch_preprocess(self, examples: list[Example]) -> list[Example]:
        # Filter out examples without public and private tests, duplicate the
        # remainder for pass@k
        examples_with_tests = [
            ex
            for ex in examples
            if len(ex["tests"]["public_tests"]) > 0
            and len(ex["tests"]["public_tests"]) + len(ex["tests"]["private_tests"]) + len(ex["tests"]["generated_tests"]) > 0
        ]
        if self.load_only_first_n is not None:
            examples_with_tests = examples_with_tests[:self.load_only_first_n]

        return batch_duplicate(examples_with_tests, self.n_samples)


    def process(self, example: Example, rng: np.random.RandomState) -> ProcessedExample:
        # Build initial prompt
        # Note: public tests are already part of the description field, so don't need to be added explicitly.

        dialog = Dialog()

        for fewshot_example in self.get_attribute_fewshot_examples:
            prompt = self.prompt_template.format(
                context=fewshot_example["description"],
                time_complexity=fewshot_example.get("time_complexity_inferred", None),
                space_complexity=fewshot_example.get("space_complexity_inferred", None),
            )
            self.add_user_message(dialog, prompt)
            # format the response !
            response = self.solution_template.format(
                code_content=fewshot_example["solution_code"]
            )
            self.add_assistant_message(dialog, response)

        prompt = self.prompt_template.format(
            context=example["description"],
            time_complexity=example.get("time_complexity_inferred", None),
            space_complexity=example.get("space_complexity_inferred", None),
        )
        self.add_user_message(dialog, prompt)
        self.add_assistant_prompt_message(dialog)

        return {"raw": example, "dialog": dialog}


    def add_response(
        self, dialog: Dialog, text, example: Example
    ) -> dict[str, float]:
        """
        Add model response to the dialog, and any additional messages required
        to carry the conversation forward.
        Modifies the dialog in-place, adding messages and updating the done flag
        """

        # Update the incomplete message with the generated tokens & hand over to user
        generation = self.update_assistant_message(dialog, text)
        dialog.done = True


    @property
    def prompt_template(self):
        match self.feedback_format:
            case "time":
                return (
                    "Provide a Python solution for the following competitive programming question: {context}.\n"
                    + 
                    "Output the code only. Generate code that has an algorithmic time complexity of {time_complexity}.\n"
                    +
                    "When analyzing the complexity of an algorithm, consider the worst-case scenario where all possible input combinations are tried, given the following conditions: 1. the inputs must adhere to the specified data types of the problem; 2. the inputs should not cause the code to crash or exit on an exception; 3. the inputs do not necessarily need to satisfy additional constraints that are potentially mentioned in the problem statement; 4. calling input() does not consume runtime nor memory, but of course any operations on top of it or afterwards will be counted towards runtime and memory footprint; 5. Anything printed gets added to the memory. You can take advantage of Python-specific optimizations provided by the underlying CPython interpreter or compiler to achieve the desired complexity, and you must account for them when analyzing the complexity.\n"
                    + 
                    self.format_guide
                )

            case "space":
                return (
                    "Provide a Python solution for the following competitive programming question: {context}.\n"
                    + 
                    "Output the code only. Generate code that has an algorithmic space complexity of {space_complexity}.\n"
                    +
                    "When analyzing the complexity of an algorithm, consider the worst-case scenario where all possible input combinations are tried, given the following conditions: 1. the inputs must adhere to the specified data types of the problem; 2. the inputs should not cause the code to crash or exit on an exception; 3. the inputs do not necessarily need to satisfy additional constraints that are potentially mentioned in the problem statement; 4. calling input() does not consume runtime nor memory, but of course any operations on top of it or afterwards will be counted towards runtime and memory footprint; 5. Anything printed gets added to the memory. You can take advantage of Python-specific optimizations provided by the underlying CPython interpreter or compiler to achieve the desired complexity, and you must account for them when analyzing the complexity.\n"
                    + 
                    self.format_guide
                )

            case "time_ranking":
                return (
                    "Provide a Python solution for the following competitive programming question: {context}.\n"
                    + 
                    "Output the code only. Generate code that has an algorithmic time complexity of {time_complexity}. Try to optimize the runtime of your code as much as you can, while respecting the time complexity requirement.\n"
                    +
                    "When analyzing the complexity of an algorithm, consider the worst-case scenario where all possible input combinations are tried, given the following conditions: 1. the inputs must adhere to the specified data types of the problem; 2. the inputs should not cause the code to crash or exit on an exception; 3. the inputs do not necessarily need to satisfy additional constraints that are potentially mentioned in the problem statement; 4. calling input() does not consume runtime nor memory, but of course any operations on top of it or afterwards will be counted towards runtime and memory footprint; 5. Anything printed gets added to the memory. You can take advantage of Python-specific optimizations provided by the underlying CPython interpreter or compiler to achieve the desired complexity, and you must account for them when analyzing the complexity.\n"
                    + 
                    self.format_guide
                )

            case "space_ranking":
                return (
                    "Provide a Python solution for the following competitive programming question: {context}.\n"
                    + 
                    "Output the code only. Generate code that has an algorithmic space complexity of {space_complexity}. Try to optimize the memory footprint of your code as much as you can, while respecting the space complexity requirement.\n"
                    +
                    "When analyzing the complexity of an algorithm, consider the worst-case scenario where all possible input combinations are tried, given the following conditions: 1. the inputs must adhere to the specified data types of the problem; 2. the inputs should not cause the code to crash or exit on an exception; 3. the inputs do not necessarily need to satisfy additional constraints that are potentially mentioned in the problem statement; 4. calling input() does not consume runtime nor memory, but of course any operations on top of it or afterwards will be counted towards runtime and memory footprint; 5. Anything printed gets added to the memory. You can take advantage of Python-specific optimizations provided by the underlying CPython interpreter or compiler to achieve the desired complexity, and you must account for them when analyzing the complexity.\n"
                    + 
                    self.format_guide
                )

            case "no_conditioning":
                return (
                    "Provide a Python solution for the following competitive programming question: {context}.\n"
                    + 
                    "Output the code only.\n"
                    + 
                    self.format_guide
                )

            case "baseline_space":
                return (
                    "Provide a Python solution for the following competitive programming question: {context}.\n"
                    + 
                    "Output the code only.\n"
                    + 
                    self.format_guide
                )

            case _:
                raise Exception('not handled !')

    @property
    def solution_template(self):
        return "```python\n{code_content}```"

    @property
    def metrics(self):
        ms = metric_names_at_k(
            at_metrics=self.at_metrics,
            n_samples=self.n_samples,
            pass_ats=self.pass_ats,
        )
        for n, k in self.pass_n_at_ks:
            ms.append(f"pass_{n}_at_{k}")
        return ms
        
    # This gets called from runner.py task_evaluation()
    def aggregate_metrics(self, metrics: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if self.n_samples == 1:
            return metrics

        agg = aggregate_at_k(
            metrics=metrics,
            at_metrics=self.at_metrics,
            n_samples=self.n_samples,
            pass_ats=self.pass_ats,
        )

        # n@k based on public test filtering
        n_filtered: dict[str, int] = defaultdict(int)
        n_correct: dict[str, int] = defaultdict(int)
        for i, raw in enumerate(metrics["raw"]):
            n_filtered[raw["task_id"]] += int(metrics["public_pass_at_1"][i] > 0)
            n_correct[raw["task_id"]] += int(metrics["pass_at_1"][i] > 0)

        for n, k in self.pass_n_at_ks:
            key = f"pass_{n}_at_{k}"
            agg[key] = []
            for task_id in n_filtered.keys():
                r = pass_n_at_k_filtered(
                    total=self.n_samples,
                    f=n_filtered[task_id],
                    c=n_correct[task_id],
                    n=n,
                    k=k,
                )
                agg[key].append(100 * r)

        return agg

    def evaluate(self, dialog: Dialog, example: Example) -> dict[str, float]:
        # Evaluate final response against public + private tests
        generation = [m for m in dialog.messages if m.source == "assistant"][-1].body
        assert generation is not None

        if "<think>" in generation and "</think>" in generation:
            generation = re.sub(f"{re.escape('<think>')}.*?{re.escape('</think>')}", "", generation, flags=re.DOTALL)

        code = self._extract_first_code(generation)

        if code is None:  # parsing error
            return {
                "pass_at_1": 0.0,
                "public_pass_at_1": 0.0,
                "compiles_at_1": 0.0,
                "code_content": "",
                "time_complexity_synthetic_ground_truth": example.get("time_complexity_inferred", None),
                "space_complexity_synthetic_ground_truth": example.get("space_complexity_inferred", None),
                "problem_name": example["problem_name"],
                "inputs_example": example["inputs_example"],
                "dataclass_code": example["dataclass_code"],
                "problem_id": example["problem_id"],
                "solution_id": example["solution_id"] + "_" + str(example["sample"]),
            }

        public_tests = example["tests"]["public_tests"]
        private_tests = example["tests"]["private_tests"]
        generated_tests = example["tests"]["generated_tests"]
        # In DMC, evaluation should done on all tests, including generated ones. See:
        # https://github.com/google-deepmind/code_contests/blob/fa7a4f8/execution/solve_example.cc#L79-L109
        tests = public_tests + private_tests + generated_tests
        assert len(tests) >= 1, "Missing test cases"
        assert len(public_tests) >= 1, "Missing public test cases"

        # assert len(tests) >= 1, "Missing test cases"
        # assert len(public_tests) >= 1, "Missing public test cases"

        results: list = exec_python_iopairs(
            source=code,
            inputs=[t["input"] for t in tests],
            outputs=[t["output"] for t in tests],
            elementwise_compare=True,
            timeout=2 + 1 * len(tests),
            early_stopping=True,
        )
        public_status = [r.status for r in results[: len(public_tests)]]
        status = [r.status for r in results]

        metrics: dict[str, float] = {}
        metrics["pass_at_1"] = 100.0 * all(s == ExecStatus.SUCCESS for s in status)
        metrics["public_pass_at_1"] = 100.0 * all(
            s == ExecStatus.SUCCESS for s in public_status
        )
        metrics["compiles_at_1"] = 100.0 * all(
            s != ExecStatus.SYNTAX_ERROR for s in status
        )
        metrics["solution_code"] = code
        metrics["time_complexity_synthetic_ground_truth"] = example.get("time_complexity_inferred", None)
        metrics["space_complexity_synthetic_ground_truth"] = example.get("space_complexity_inferred", None)
        metrics["problem_name"] = example["problem_name"]
        metrics["inputs_example"] = example["inputs_example"]
        metrics["dataclass_code"] = example["dataclass_code"]
        metrics["problem_id"] = example["problem_id"]
        metrics["solution_id"] = example["solution_id"] + "_" + str(example["sample"])

        return metrics

    @staticmethod
    def _extract_first_code(text: str):
        # Try ```python <code> ``` first since we specifically asked for it;
        # match any triple backticks otherwise.
        pattern_py = r"``` *(python|py)(.*?)```"
        pattern = r"``` *(.*?)```"
        if match := re.search(pattern_py, text, re.DOTALL):
            return match.group(2)
        if match := re.search(pattern, text, re.DOTALL):
            return match.group(1)
        return None

class FullComplexityGenerationTask(ComplexityGenerationTask):
    load_only_first_n = None

class TinyComplexityGenerationTask(ComplexityGenerationTask):
    load_only_first_n = 10


ALL_PROMPT_VARIANTS = [*PROMPT_VARIANTS]

ALL_PASS_ATS = {"": [1], **DEFAULT_PASS_ATS}


for prompt_variant, max_attempts, single_shot, (at_k, pass_at), use_fewshot_local, use_small_dataset in product(
    ALL_PROMPT_VARIANTS, [None,], [False,], ALL_PASS_ATS.items(), [False], [False, True]
):
    task_suffixes = []
    if prompt_variant is not None:
        if prompt_variant == "time_ranking":
            task_suffixes.append(f"time")
        elif prompt_variant == "space_ranking":
            task_suffixes.append(f"space")
        else:
            task_suffixes.append(f"{prompt_variant}")

    if use_small_dataset:
        task_suffixes.append("tiny")

    if use_fewshot_local:
        task_suffixes.append("fewshot")

    if max_attempts is not None:
        # we want 1
        task_suffixes.append(f"max{max_attempts}")

    if single_shot:
        # we want False
        task_suffixes.append("single_shot")

    if len(at_k) > 0:
        # we want at_10
        task_suffixes.append(f"{at_k}")

    if len(task_suffixes) > 0:
        task_suffix = "_".join(task_suffixes)

    else:
        task_suffix = ""

    if (f"complexity_generation/{task_suffix}" not in TaskRegistry.names()) and "ranking" not in prompt_variant:
        if prompt_variant == "time":
            if use_small_dataset:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        TinyComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_time_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
            else:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        FullComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_time_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
        elif prompt_variant == "space":
            if use_small_dataset:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        TinyComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_space_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
            else:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        FullComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_space_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
        elif prompt_variant == "no_conditioning":
            if use_small_dataset:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        TinyComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file=None,
                    ),
                )
            else:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        FullComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file=None,
                    ),
                )
        elif prompt_variant == "baseline_space":
            if use_small_dataset:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        TinyComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file=None,
                    ),
                )
            else:
                TaskRegistry.register(
                    f"complexity_generation/{task_suffix}",
                    partial(
                        FullComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file=None,
                    ),
                )
        else:
            raise Exception('not handled')

    elif (f"complexity_ranking/{task_suffix}" not in TaskRegistry.names()) and "ranking" in prompt_variant:
        if prompt_variant == "time_ranking":
            if use_small_dataset:
                TaskRegistry.register(
                    f"complexity_ranking/{task_suffix}",
                    partial(
                        TinyComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_time_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
            else:
                TaskRegistry.register(
                    f"complexity_ranking/{task_suffix}",
                    partial(
                        FullComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_time_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
        elif prompt_variant == "space_ranking":
            if use_small_dataset:
                TaskRegistry.register(
                    f"complexity_ranking/{task_suffix}",
                    partial(
                        TinyComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_space_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
            else:
                TaskRegistry.register(
                    f"complexity_ranking/{task_suffix}",
                    partial(
                        FullComplexityGenerationTask,
                        pass_ats=pass_at,
                        max_attempts=max_attempts,
                        single_shot=single_shot,
                        feedback_format=prompt_variant,
                        use_fewshot_local=use_fewshot_local,
                        fewshot_file="fewshot_space_evaluation.jsonl" if use_fewshot_local else None,
                    ),
                )
        else:
            raise Exception('not handled')

    else:
        raise Exception('not handled')