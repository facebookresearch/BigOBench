# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import (
    Any,
    NoReturn,
    Optional,
    TypeVar,
)
from src.eval.iterator.dialog import (
    MessageV2,
)


@dataclasses.dataclass
class Packet:
    thread_id: Any
    "An arbitrary identifier to link inputs to outputs."
    tokens: Optional[list[int]] = dataclasses.field(default_factory=list)
    dialog: Optional[list[MessageV2]] = None
    "The prompt if used as input, or else the generation output."
    text: Optional[str] = None
    "The prompt given as string, used if ``tokens`` is empty."
    temperature: Optional[float] = None
    "An optional per-generation temperature setting."
    logprobs: Optional[list[float]] = None
    """
    Logprobs for the tokens list. In output packets only,
    and only when requested.
    """

NoErr = NoReturn

TErr = TypeVar("TErr")