# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from os import path as osp


class ExecStatus(Enum):
    UNKNOWN = -1
    SUCCESS = 0
    FAILURE = 1
    EXCEPTION = 2
    SYNTAX_ERROR = 3
    TIMEOUT = 4


@dataclass
class ExecResult:
    status: ExecStatus = ExecStatus.UNKNOWN
    info: str = ""


_sandbox_dir = osp.join(osp.dirname(osp.abspath(__file__)), "sandbox")


def sandbox_dir():
    return _sandbox_dir
