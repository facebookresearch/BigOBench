# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional
from os import path as osp
import traceback
import time
import logging
from . import ExecResult
import sys
import os


logger = logging.getLogger()


def _get_python_time_space_complexities(
    output_r,  # JSONConnection,
    early_stopping: bool,
    global_timeout: int,
):
    """
    Retrieves runtime and memory usage data from a remote connection.
    Args:
        output_r (JSONConnection): A connection object to receive data from.
        early_stopping (bool): Whether to stop early if an error occurs. (Currently not used in this function)
        global_timeout (int): The maximum time in seconds to wait for a response.
    Returns:
        tuple: A tuple containing four lists:
            - runtime_dict_list (list): A list of dictionaries containing runtime data.
            - memory_dict_list (list): A list of dictionaries containing memory usage data.
            - error_list (list): A list of errors that occurred during data retrieval.
            - debugging_dict_list (list): A list of dictionaries containing debugging information.
    Raises:
        Exception: If a timeout occurs or an error is encountered while receiving data.
    """

    runtime_dict_list = []
    memory_dict_list = []
    error_list = []
    debugging_dict_list = []

    start_time = time.perf_counter()
    try:
        while True:
            if not output_r.poll(timeout=start_time + global_timeout - time.perf_counter()):
                raise Exception('time out')

            response = output_r.recv()

            if "runtime_dict_list" in response:
                runtime_dict_list = response["runtime_dict_list"]
                memory_dict_list = response["memory_dict_list"]
                error_list = response["error_list"]
                debugging_dict_list = response["debugging_dict_list"]
                break

            elif "error" in response:
                error_list.append(response["error"])
                break

    except Exception as e:
        error_list.append(e)

    return runtime_dict_list, memory_dict_list, error_list, debugging_dict_list


def exec_python_time_space_complexities(
    question_inputs_list: List[dict],
    multiplier_start: float,
    multiplier_repeat: float,
    multiplier_end: float,
    multiplier_mult_step: float,
    multiplier_max_increase: float,
    time_profiler: str,
    print_debugging_dict_list: bool,
    correct_nlogn: bool,
    multiplier_op: str,
    shuffle_runs: bool,
    timeout: int,
    large_timeout: int,
    giga_timeout: int,
    global_timeout: int,
    memory_limit: int,
    size_of_other_arguments: int,
    temp_file_name_seed: str,
    fork_server=None,  # Optional[ForkServer] = None,
    bubblewrap_executor=None,  # Executor = Executor.BUBBLEWRAP,
    cpu_id_list: Optional[List[int]] = None,
    early_stopping: bool = False,
    forkserver_type: str = 'standard',
    input_handler: str = 'with_dataclass',
    **spawn_args,
) -> List[ExecResult]:
    """
    Execute Python code to measure time and space complexities.
    This function spawns a new process using a fork server, sends the input data to the process,
    and receives the results. It measures the time and space complexities of the executed code.
    Args:
        question_inputs_list (List[dict]): A list of dictionaries containing input data.
        multiplier_start (float): The starting value of the multiplier.
        multiplier_repeat (float): The repeat value of the multiplier.
        multiplier_end (float): The ending value of the multiplier.
        multiplier_mult_step (float): The step value of the multiplier.
        multiplier_max_increase (float): The maximum increase value of the multiplier.
        time_profiler (str): The type of time profiler to use.
        print_debugging_dict_list (bool): Whether to print debugging information.
        correct_nlogn (bool): Whether to correct for n log n complexity.
        multiplier_op (str): The operation to perform on the multiplier.
        shuffle_runs (bool): Whether to shuffle the runs.
        timeout (int): The timeout value in seconds.
        large_timeout (int): The large timeout value in seconds.
        giga_timeout (int): The giga timeout value in seconds.
        global_timeout (int): The global timeout value in seconds.
        memory_limit (int): The memory limit in bytes.
        size_of_other_arguments (int): The size of other arguments.
        temp_file_name_seed (str): The seed for temporary file names.
        fork_server (Optional[ForkServer], optional): The fork server to use. Defaults to None.
        bubblewrap_executor (Optional[Executor], optional): The bubblewrap executor to use. Defaults to None.
        cpu_id_list (Optional[List[int]], optional): The list of CPU IDs to use. Defaults to None.
        early_stopping (bool, optional): Whether to stop early. Defaults to False.
        forkserver_type (str, optional): The type of fork server to use. Defaults to 'standard'.
        input_handler (str, optional): The input handler to use. Defaults to 'with_dataclass'.
        **spawn_args: Additional keyword arguments to pass to the spawn method.
    Returns:
        List[ExecResult]: A list of execution results.
    """

    if forkserver_type == 'standard':
        from sandbox.server_runtimes import ForkServer, JSONConnection, Executor
    elif forkserver_type == 'custom':
        from sandbox.server_runtimes_custom import Executor, ForkServer, JSONConnection
    else:
        raise Exception('forkserver type not handled')

    if bubblewrap_executor is None:
        bubblewrap_executor = Executor.BUBBLEWRAP

    if fork_server is None:
        fork_server = ForkServer.global_instance()

    vpid, input_w, output_r = fork_server.spawn(
        cmd=[sys.executable, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "worker.py")],
        cpu_id_list=cpu_id_list,
        executor=bubblewrap_executor,
        **spawn_args,
    )

    try:
        start_time = time.time()
        input_w.send(
            {
                "question_inputs_list": question_inputs_list,
                "multiplier_start": multiplier_start,
                "multiplier_repeat": multiplier_repeat,
                "multiplier_end": multiplier_end,
                "multiplier_mult_step": multiplier_mult_step,
                "multiplier_max_increase": multiplier_max_increase,
                "time_profiler": time_profiler,
                "print_debugging_dict_list": print_debugging_dict_list,
                "correct_nlogn": correct_nlogn,
                "multiplier_op": multiplier_op,
                "cpu_id_list": cpu_id_list,
                "timeout": timeout,
                "large_timeout": large_timeout,
                "giga_timeout": giga_timeout,
                "memory_limit": memory_limit,
                "size_of_other_arguments": size_of_other_arguments,
                "shuffle_runs": shuffle_runs,
                "temp_file_name_seed": temp_file_name_seed,
                "input_handler": input_handler,
            }
        )
        return _get_python_time_space_complexities(output_r, early_stopping, global_timeout)

    finally:
        # print(vpid, time.time() - start_time)
        input_w.close()
        output_r.close()

        if vpid is not None:
            fork_server.kill(vpid)


def _get_test_results(
    output_r, #JSONConnection,
    num_tests: int,
    early_stopping: bool,
    timeout: float,
):
    from sandbox.server import ForkServer, JSONConnection, Executor
    from . import ExecStatus

    test_results = [ExecResult()] * num_tests
    start_time = time.perf_counter()
    try:
        while True:
            if not output_r.poll(timeout=start_time + timeout - time.perf_counter()):
                for i in range(len(test_results)):
                    if test_results[i].status == ExecStatus.UNKNOWN:
                        test_results[i] = ExecResult(status=ExecStatus.TIMEOUT)
                break

            res = output_r.recv()

            if "test_case" in res:
                if res["result"] == "pass":
                    test_results[res["test_case"]] = ExecResult(
                        status=ExecStatus.SUCCESS
                    )
                elif res["result"] == "failure":
                    test_results[res["test_case"]] = ExecResult(
                        status=ExecStatus.FAILURE,
                        info=res["exception"],
                    )
                    if early_stopping:
                        break
                elif res["result"] == "exception":
                    test_results[res["test_case"]] = ExecResult(
                        status=ExecStatus.EXCEPTION,
                        info=res["exception"],
                    )
                    if early_stopping:
                        break
                else:
                    pass
            elif "error" in res:
                if "SyntaxError" in res["error"] or "IndentationError" in res["error"]:
                    test_results = [
                        ExecResult(status=ExecStatus.SYNTAX_ERROR, info=res["error"])
                    ] * num_tests
                else:
                    test_results = [
                        ExecResult(status=ExecStatus.EXCEPTION, info=res["error"])
                    ] * num_tests
                break
            elif "done" in res:
                break
    except BaseException as e:
        logger.exception(
            f"Error parsing execution results: {e}\n{traceback.format_exc()}"
        )
        test_results = [ExecResult()] * num_tests

    return test_results


def exec_python_iopairs(
    source: str,
    inputs: List[str],
    outputs: List[str],
    strip_output: bool = True,
    elementwise_compare: bool = False,
    timeout: float = 60.0,
    early_stopping: bool = False,
    fork_server = None, #Optional[ForkServer] = None,
    **spawn_args,
) -> List[ExecResult]:
    from sandbox.server import ForkServer, JSONConnection, Executor

    if fork_server is None:
        fork_server = ForkServer.global_instance()

    vpid, input_w, output_r = fork_server.spawn(
        cmd=[sys.executable, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "worker_wo_measures.py")],
        **spawn_args,
    )

    try:
        input_w.send(
            {
                "source": source,
                "inputs": inputs,
                "outputs": outputs,
                "strip_output": strip_output,
                "elementwise_compare": elementwise_compare,
            }
        )
        return _get_test_results(output_r, len(inputs), early_stopping, timeout)
    finally:
        input_w.close()
        output_r.close()
        fork_server.kill(vpid)

