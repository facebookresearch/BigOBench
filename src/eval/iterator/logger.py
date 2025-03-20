# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
import sys
import time
import logging
import math

class LogFormatter(logging.Formatter):
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        # define prefix
        # record.pathname / record.filename / record.lineno
        subsecond, seconds = math.modf(record.created)
        curr_date = (
            time.strftime("%y-%m-%d %H:%M:%S", time.localtime(seconds))
            + f".{int(subsecond * 1_000_000):06d}"
        )
        delta = timedelta(seconds=round(record.created - self.start_time))

        prefix = f"{record.levelname:<7} {curr_date} - {delta} - "

        # logged content
        content = record.getMessage()
        indent = " " * len(prefix)
        content = content.replace("\n", "\n" + indent)

        # Exception handling as in the default formatter, albeit with indenting
        # according to our custom prefix
        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)

        if record.exc_text:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            content = content + indent.join(
                [l + "\n" for l in record.exc_text.splitlines()]
            )
            if content[-1:] == "\n":
                content = content[:-1]
        if record.stack_info:
            if content[-1:] != "\n":
                content = content + "\n" + indent
            stack_text = self.formatStack(record.stack_info)
            content = content + indent.join([l + "\n" for l in stack_text.splitlines()])
            if content[-1:] == "\n":
                content = content[:-1]

        return prefix + content


def initialize_logger(log_level: str = "NOTSET") -> logging.Logger:
    set_root_log_level(log_level)
    logger = logging.getLogger()

    # stdout: everything
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.NOTSET)
    stdout_handler.setFormatter(LogFormatter())

    # stderr: warnings / errors and above
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(LogFormatter())

    # set stream handlers
    logger.handlers.clear() 
    assert len(logger.handlers) == 0, logger.handlers
    logger.handlers.append(stdout_handler)
    logger.handlers.append(stderr_handler)

    return logger


def add_logger_file_handler(filepath: str):
    # build file handler
    file_handler = logging.FileHandler(filepath, "a")
    file_handler.setLevel(logging.NOTSET)
    file_handler.setFormatter(LogFormatter())

    # update logger
    logger = logging.getLogger()
    logger.addHandler(file_handler)


def set_root_log_level(log_level: str):
    logger = logging.getLogger()
    level: int | str = log_level.upper()
    try:
        level = int(log_level)
    except ValueError:
        pass

    try:
        logger.setLevel(level)  # type: ignore
    except Exception:
        logger.warning(
            f"Failed to set logging level to {log_level}, using default 'NOTSET'"
        )
        logger.setLevel(logging.NOTSET)
