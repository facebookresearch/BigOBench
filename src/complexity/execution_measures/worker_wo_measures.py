# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Runner for Python code and a variable number of input/output pairs.
# For each pair, input data is sent to stdin and data written to stdout is
# compared to the expected output data.
#
# NOTE: invocations for different input/output pairs are not fully isolated from
# each other; for example, modules are loaded only once per Python process.
# If this turns out to cause trouble, a simple fix would be to fork into separate
# processes per input/output pair.
#

import builtins
import json
import linecache
import re
import sys
import traceback
import decimal
from io import StringIO
from multiprocessing.connection import Connection
from typing import List
from unittest.mock import mock_open


class EarlyReturnException(Exception):
    """Custom exception to handle early returns in user code."""

    pass


def custom_exit(*args, **kwargs):
    """Function to replace exit() and quit(), raises EarlyReturnException."""
    raise EarlyReturnException()


def execute_code(code):
    """Execute wrapper to handle EarlyReturnException and if __name__ == '__main__'"""
    try:
        # Execute user code with __name__ set to '__main__'
        exec(code, {"__name__": "__main__"})
    except EarlyReturnException:
        # Early return from user code
        pass


def try_dec(x: str):
    try:
        return True, decimal.Decimal(x)
    except decimal.InvalidOperation:
        return False, decimal.Decimal()


def compare_val(xa: str, xb: str):
    xa_isdec, xa_d = try_dec(xa)
    xb_isdec, xb_d = try_dec(xb)
    if xa_isdec and xb_isdec:
        return (xa_d - xb_d).copy_abs() < 1e-5
    return xa == xb


def compare(a: str, b: str, elementwise: bool):
    if not elementwise:
        return a == b

    # https://github.com/google-deepmind/code_contests/blob/fa7a4f8/execution/tester_sandboxer.cc#L256
    va = [x.lower() for x in re.split(r"[ \n\t\r\v]", a) if x]
    vb = [x.lower() for x in re.split(r"[ \n\t\r\v]", b) if x]
    if va == vb:
        return True
    if len(va) != len(vb):
        return False

    for xa, xb in zip(va, vb):
        if not compare_val(xa, xb):
            return False
    return True


def main() -> None:
    input_r = Connection(int(sys.argv[1]), writable=False)
    output_w = Connection(int(sys.argv[2]), readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    data = input_r.recv()

    source: str = data["source"]
    inputs: List[str] = data["inputs"]
    outputs: List[str] = data["outputs"]
    strip_output: bool = data["strip_output"]
    elementwise_compare: bool = data["elementwise_compare"]
    if len(inputs) != len(outputs):
        output_w.send_bytes(
            json.dumps(
                {"error": f"Got {len(inputs)} inputs but {len(outputs)} outputs"}
            ).encode("utf8")
        )
        return

    # Pre-compile for faster execution
    try:
        compiled = compile(source, "<source>", "exec")
        linecache.cache["<source>"] = (
            len(source),
            None,
            source.splitlines(True),
            "<source>",
        )
    except BaseException:
        output_w.send_bytes(
            json.dumps({"error": traceback.format_exc()}).encode("utf8")
        )
        return

    for (
        i,
        (inp, outp),
    ) in enumerate(zip(inputs, outputs)):
        sys.stdin = StringIO(inp)
        sys.stdout = StringIO()
        builtins.exit = custom_exit
        builtins.quit = custom_exit
        sys.exit = custom_exit
        # deal with the case of open(0).read()
        builtins.open = mock_open(read_data=inp)
        try:
            execute_code(compiled)

            out = sys.stdout.getvalue()
            if strip_output:
                out = out.strip()
                outp = outp.strip()
            if compare(out, outp, elementwise_compare):
                output_w.send_bytes(
                    json.dumps(
                        {
                            "test_case": i,
                            "result": "pass",
                        }
                    ).encode("utf8")
                )
            else:
                output_w.send_bytes(
                    json.dumps(
                        {
                            "test_case": i,
                            "result": "failure",
                            "exception": f"Expected output `{outp}` but got `{out}`",
                        }
                    ).encode("utf8")
                )
        except BaseException:
            output_w.send_bytes(
                json.dumps(
                    {
                        "test_case": i,
                        "result": "exception",
                        "exception": traceback.format_exc(),
                    }
                ).encode("utf8")
            )


if __name__ == "__main__":
    main()
