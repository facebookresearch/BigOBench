# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Any, Type, Optional, Sequence, TypeAlias, Union

import numpy as np

from src.eval.iterator.dialog import (
    Message,
    MessageBase,
    MessageV2,
    SampleSFT,
    convert_dialog_message_v1_to_message_v2,
)
from src.eval.task.utils import (
    aggregate_at_k,
    get_n_samples_for_pass_ats,
    metric_names_at_k,
)


Example: TypeAlias = Dict[str, Any]
ProcessedExample: TypeAlias = Dict[str, Any]
Prompt = Union[str, SampleSFT]

# Options for fewshot mode: first, random and index
# first: select the first n_fewshot examples of the fewshot file as fewshot examples for all test examples
# random: select n_fewshot random examples from the fewshot file as fewshot examples, examples similar to the current test example will be discarded
# index: select fewshot examples in the fewshot file at the index specific by fewshot_index list
# for instance fewshot_index = [1, 4] will use the second and fifth examples as fewshot examples for all examples.
class FewshotMode(Enum):
    FIRST = "first"
    RANDOM = "random"
    INDEX = "index"


class BaseTask:
    """
    Base task for all evaluation tasks.
    """

    # List of metrics
    metrics: List
    # Allows files post-assignment, if set to True don't check valid paths at init time
    allow_post_init_dataset_assignment: bool = False

    data_file_path: str

    # Fewshot evaluation support
    fewshot_file: str
    fewshot_mode: FewshotMode
    n_fewshot: int = 0
    fewshot_index: List[int] = []

    load_only_first_n: int = None


    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        if not self.allow_post_init_dataset_assignment:
            assert self.data_file_path.endswith(".jsonl")
            assert os.path.isfile(self.data_file_path), f"Eval file does not exist {self.data_file_path}"

        # Initialize fewshot examples
        self._all_fewshot_examples: List[Example] = []
        if self.n_fewshot > 0 or len(self.fewshot_index) > 0:
            assert self.fewshot_file.endswith(".jsonl")
            assert os.path.isfile(
                self.fewshot_file
            ), f"Fewshot file does not exist: {self.fewshot_file}"

    def _load_fewshot_file(self) -> List[Example]:
        all_fewshot_examples: List[Example] = []
        if self.n_fewshot > 0 or len(self.fewshot_index) > 0:
            with open(
                self.fewshot_file
            ) as fin:
                for line in fin:
                    if not line:
                        continue
                    all_fewshot_examples.append(json.loads(line))
        return all_fewshot_examples

    def get_fewshot_examples(
        self, example: Example, rng: np.random.RandomState
    ) -> List[Example]:
        """Get fewshot examples for the given example leveraging the appropriate fewshot mode."""
        if self.n_fewshot == 0 and len(self.fewshot_index) == 0:
            raise ValueError(
                "It's not possible to get fewshot examples with these attibute values"
            )

        if len(self._all_fewshot_examples) == 0:
            self._all_fewshot_examples = self._load_fewshot_file()

        if self.fewshot_mode == FewshotMode.INDEX:
            fewshot_examples = [
                self._all_fewshot_examples[k] for k in self.fewshot_index
            ]
        elif self.fewshot_mode == FewshotMode.FIRST:
            fewshot_examples = self._all_fewshot_examples[: self.n_fewshot]
        elif self.fewshot_mode == FewshotMode.RANDOM:
            while True:
                indices = rng.choice(
                    range(len(self._all_fewshot_examples)),
                    self.n_fewshot,
                    replace=False,
                )
                fewshot_examples = [self._all_fewshot_examples[idx] for idx in indices]
                if all(example != shot for shot in fewshot_examples):
                    break
        else:
            raise ValueError(f"Fewshot strategy {self.fewshot_mode} not supported")
        return fewshot_examples

    def aggregate_metrics(self, metrics: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Method to aggregate metrics after evaluation."""
        return metrics

    @abstractmethod
    def process(self, example: Example, rng: np.random.RandomState) -> ProcessedExample:
        raise RuntimeError("not implemented")

    def batch_preprocess(self, examples: List[Example]) -> List[Example]:
        """Method to transform all raw examples at the beginning of the TaskIterator."""
        return examples


@dataclass
class Dialog:
    messages: List[MessageBase] = field(default_factory=list)
    # Tokens presented to and generated by the model
    message_tokens: List[List[int]] = field(default_factory=list)
    # Indicates whether a token was generated by the model
    message_tokens_generated: List[List[bool]] = field(default_factory=list)
    # Any state the task needs to maintain per dialog
    data: Optional[Any] = None
    # Whether this dialog is done
    done: bool = False


class ChatBaseTaskMixin:
    """
    Base task for all finetuning chat-format evaluations.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*[*args], **kwargs)
        # chat tasks expect an InstructTokenizer
        self.use_message_v2 = True

    def create_prompt_from_dialog(self, dialog: Sequence[MessageBase]) -> "SampleSFT":
        # convenient function to create a SampleSFT from dialog for either Message or MessageV2
        # to be used by children chat tasks
        if self.use_message_v2:
            dialog_v2 = convert_dialog_message_v1_to_message_v2(dialog)  # type: ignore
            return SampleSFT(dialog=dialog_v2)
        else:
            return SampleSFT(dialog=dialog)

    def create_prompt_with_fewshot_from_dialog(
        self, prompt: Prompt, fewshot_dialog: Sequence[MessageBase]
    ) -> "SampleSFT":
        # convenient function to create a SampleSFT from dialog for either Message or MessageV2
        # in the case of fewshot examples where we combine an already-formatted SampleSFT prompt
        # and fewshot examples as a dialog of messages.
        assert isinstance(prompt, SampleSFT)
        if self.use_message_v2:
            fewshot_dialog = convert_dialog_message_v1_to_message_v2(fewshot_dialog)  # type: ignore
        prompt_with_fewshot = SampleSFT(dialog=fewshot_dialog)
        prompt_with_fewshot.dialog.extend(prompt.dialog)  # type: ignore
        return prompt_with_fewshot


class MultiturnTask(BaseTask):
    # Set this to true if evaluate() is thread-safe.  This will allow the
    # evaluation loop to parallelize this calls for higher decoding throughput.
    has_threadsafe_evaluate: bool = False

    @abstractmethod
    def process(self, example: Example, rng: np.random.RandomState) -> ProcessedExample:
        # ProcessedExample is expected to have a new Dialog instance at "dialog"
        pass

    @abstractmethod
    def get_prompt_tokens(self, dialog: Dialog) -> List[int]:
        pass

    @abstractmethod
    def add_response(
        self, dialog: Dialog, text: str, example: Example
    ) -> dict[str, float]:
        # add model response to the dialog, and any additional messages required
        # to carry the conversation forward.
        # updated dialog and dialog.done to true/false depending on whether the
        # dialog ended.
        # return metrics.
        pass

    @abstractmethod
    def evaluate(self, dialog: Dialog, example: Example) -> Dict[str, float]:
        # get eval metrics for a completed -- or aborted -- dialog
        pass


class ChatMultiturnTask(MultiturnTask):
    """
    A multiturn task with convenience functions for dialog handling in
    Message/Chat formats.
    Model generations are from the "assistant" source, other messages (besides
    "system") are from "user".
    """

    def __init__(
        self,
        data_file_path: str,
        message_cls: type = MessageV2,
    ):
        super().__init__(data_file_path)
        self.message_cls = message_cls

    def get_prompt_tokens(self, dialog: Dialog) -> list[int]:
        return dialog.messages
    
    def update_assistant_message(self, dialog: Dialog, text: str, ):
        assert dialog.messages[-1].source == "assistant"

        if dialog.messages[-1].body is None:
            dialog.messages[-1].body = text
        else:
            dialog.messages[-1].body += text

        dialog.messages[-1].eot = True

        return text

    def add_system_prompt_message(self, dialog: Dialog, text: str, bos=True):
        m = self.message_cls.system(body=text)
        dialog.messages.append(m)  # type: ignore

    def add_user_message(self, dialog: Dialog, text: str):
        m = self.message_cls.user(text)  # type: ignore
        dialog.messages.append(m)  # type: ignore
        
    def add_assistant_message(self, dialog: Dialog, text: str):
        m = self.message_cls.assistant(text)
        dialog.messages.append(m)

    def add_assistant_prompt_message(self, dialog: Dialog, text: str | None = None):
        m = self.message_cls.assistant(body=text)
        dialog.messages.append(m)

    def num_assistant_messages(self, dialog: Dialog):
        return len([m for m in dialog.messages if m.source == "assistant"])
