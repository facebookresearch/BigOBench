# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
from json import JSONDecodeError
from logging import getLogger
from typing import Dict, Iterator, List, Optional, Tuple
import gzip
import numpy as np
from functools import partial

SequenceWithMask = Tuple[List[int], List[bool]]


BEGIN_INST_TAG = "[INST]"
END_INST_TAG = "[/INST]"
BEGIN_FILENAME_TAG = " [FILENAME]"

logger = getLogger()


class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        open_func = (
            partial(gzip.open, fpath, "rt", encoding="utf-8")  # type: ignore
            if fpath.endswith(".jsonl.gz")
            else partial(open, fpath, "r", encoding="utf-8")  # type: ignore
        )
        self.f = open_func()
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen(infinite))
        self.iter_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def _post_process_json(self, json):
        return json

    def gen(self, infinite: bool) -> Iterator[Dict]:
        while True:
            logger.info(f"Starting iteration {self.iter_id} over {self.fpath} ...")
            self.iter_id += 1
            while True:
                line, self.line_num = self.f.readline(), self.line_num + 1
                if not line:
                    break
                if (self.line_num - 1) % self.world_size == self.world_rank:
                    try:
                        yield self._post_process_json(json.loads(line))
                    except JSONDecodeError as e:
                        logger.error(
                            f"Error parsing line {self.line_num}: '{line}' from {self.fpath}"  # type: ignore
                        )
                        raise e
            if not infinite:
                break
            self.set_position(None)
        self.f.close()

    def set_position(self, position: Optional[int]):
        logger.warning(
            f"Setting JSONL position on {self.fpath} "
            f"({self.world_rank}/{self.world_size}): {position}"
        )
        if position is None:
            self.f.seek(0)
            self.line_num = 0
        else:
            assert type(position) is int
            self.f.seek(position)
            self.line_num = (
                self.world_rank + 1
            )  # restore value of line_num (modulo world_size)

    def get_position(self) -> Optional[int]:
        file_pos = self.f.tell()
        if file_pos == 0 and self.line_num == 0:
            return None
        assert (self.line_num - 1) % self.world_size == self.world_rank
        return file_pos

