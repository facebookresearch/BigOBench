# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import faulthandler
import json
import logging
import signal
import sys
import resource
import traceback
from io import StringIO
from multiprocessing.connection import Connection
from typing import Any, Dict
import builtins
from unittest.mock import mock_open
import json
import numpy as np
import timeit
import re
import os
import linecache
import subprocess
import sys
import tempfile
import signal
import time
from multiprocessing.connection import Connection
from string import Template
import string
import time
import cProfile
import random
import copy
import psutil
import collections
import gc
from os.path import abspath

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from input_generations.atomic_types import *
from input_generations.input_expansion_methods import *
from input_generations.one_dim_list_types import *
from input_generations.tuple_list_types import *
from input_generations.two_dim_list_types import *

from worker_utils import (
    # used to preprocess arbitrary code so that it is better executable
    preprocess_code_content,
    # used to find the executable function to evaluate in a piece of code
    wrap_code_to_get_executable_function,
    add_context_to_root_functions,  # used to better handle imports in code
    # replace tim sort with a more predictable sorting algorithm (merge sort)
    replace_sorting_algorithm,
    make_root_functions_accessible_inside_classes,  # improve code executability
    make_root_objects_accessible_inside_classes,
    add_context_to_root_objects,
    execute_code_with_inputs,  # to import the values of the inputs in the code
    generate_expansion_details_list,
    get_import_statements,
    str_variable_name_type_dimension,
    get_source_tuple,
    get_source_tuple_with_dataclass,
    get_variable_name_to_input_dict_stringified,
)


def main() -> None:
    """
    Main Function

    This function serves as the entry point for executing a complex Python program designed to handle
    various computational tasks, including code execution, input handling, and performance profiling.
    The function is structured to manage resources, handle inputs, execute code, and collect performance
    metrics. Below is a detailed step-by-step explanation of the function's operations:

    1. **Resource Limitation**:
    - `resource.setrlimit(resource.RLIMIT_NPROC, (200, 200))`: Limits the number of processes to prevent forkbombs.

    2. **Connection Setup**:
    - Establishes input and output connections using `Connection` objects to receive and send data.

    3. **Initial Setup**:
    - Sends a canary message to verify the connection.
    - Seeds the random number generators for `random` and `numpy` to ensure reproducibility.

    4. **Data Reception**:
    - Receives data from the input connection, which includes various parameters and configurations for execution.

    5. **CPU Affinity**:
    - Sets and verifies CPU affinity for the current process if specified in the input data.

    6. **Parameter Extraction**:
    - Extracts various parameters from the input data, such as timeouts, memory limits, and input handlers.

    7. **Import Statements**:
    - Retrieves common import statements using `get_import_statements` to be used in dynamically compiled code.

    8. **Memory Limitation**:
    - Sets memory limits for the process to prevent excessive memory usage.

    9. **Signal Handling**:
    - Defines a signal handler to manage timeouts and interrupts during execution.

    10. **Template and Replacement Setup**:
        - Sets up templates and replacement pairs for input processing and code generation.

    11. **Standard I/O Redirection**:
        - Redirects standard input and output to `StringIO` objects to capture and manipulate I/O during execution.

    12. **Input Parsing**:
        - Parses and processes input data, including handling dataclass inputs if specified.

    13. **Variable Type and Dimension Detection**:
        - Detects the types and dimensions of input variables using dynamic code execution and templates.

    14. **Executable Code Preparation**:
        - Prepares the executable code by wrapping it in a context and handling root functions and objects.

    15. **Sorting Algorithm Replacement**:
        - Replaces the built-in sorting algorithm with a custom merge sort if specified.

    16. **Input Precomputation**:
        - Precomputes inputs for various types and dimensions to optimize execution time.

    17. **Execution Details Registration**:
        - Registers execution details, including input variations and configurations, for later execution.

    18. **Execution and Profiling**:
        - Executes the prepared code with various inputs and profiles its performance using specified profilers.

    19. **Error Handling**:
        - Catches and logs errors during execution and profiling, ensuring robust error reporting.

    20. **Output Sending**:
        - Sends the collected runtime, memory, and debugging information back through the output connection.

    21. **Exception Handling**:
        - Catches any exceptions that occur during the entire process and sends error information through the output connection.

    The function is designed to handle complex computational tasks with a focus on resource management,
    input handling, and performance profiling. It uses a combination of dynamic code execution, input
    preprocessing, and profiling techniques to achieve its goals.
    """
    
    # limit the damage that can be done by forkbombs
    resource.setrlimit(resource.RLIMIT_NPROC, (200, 200))

    input_r = Connection(int(sys.argv[1]), writable=False)
    output_w = Connection(int(sys.argv[2]), readable=False)
    output_w.send_bytes(json.dumps({"canary": "chirp"}).encode("utf8"))

    random.seed(0)
    np.random.seed(0)

    try:
        data = input_r.recv()

        if data["cpu_id_list"] is not None:
            cpu_id_list = data["cpu_id_list"]
            current_process = psutil.Process(os.getpid())
            # current_process.cpu_affinity(cpu_id)
            try:
                assert current_process.cpu_affinity() == cpu_id_list
            except:
                raise Exception(str((current_process.cpu_affinity(), cpu_id_list)))

        else:
            cpu_id_list = None
            current_process = psutil.Process(os.getpid())

        question_inputs_list = data["question_inputs_list"]
        multiplier_start = int(data["multiplier_start"])
        multiplier_repeat = int(data["multiplier_repeat"])
        multiplier_end = int(data["multiplier_end"])
        multiplier_mult_step = int(data["multiplier_mult_step"])
        multiplier_max_increase = int(data["multiplier_max_increase"])
        print_debugging_dict_list = data["print_debugging_dict_list"]
        input_handler = data["input_handler"]

        correct_nlogn = bool(data["correct_nlogn"])

        multiplier_op = data["multiplier_op"]
        time_profiler = data["time_profiler"]

        timeout = int(data["timeout"])  # 1
        large_timeout = int(data["large_timeout"])  # 1
        giga_timeout = int(data["giga_timeout"])  # 5
        size_of_other_arguments = int(data["size_of_other_arguments"])  # 1000

        shuffle_runs = bool(data["shuffle_runs"])
        temp_file_name_seed = str(data["temp_file_name_seed"])

        assert time_profiler in ['time', 'cprofile', 'timeit',
                                 'cprofilewithin', 'cprofilearound', 'cprofilerobust']
        assert multiplier_op in ['addition', 'multiplication']

        if multiplier_op == "multiplication":
            def multiplier_op(x, y): return x * y
        elif multiplier_op == "addition":
            def multiplier_op(x, y): return x + y
        else:
            raise Exception('not supported')

        import_statements = get_import_statements(correct_nlogn)

        (
            variable_type_dimension_method_to_base_multiplier_priority,
            variable_type_dimension_to_expansion_details_list,
        ) = get_expansion_details()

        runtime_dict_list = []
        debugging_dict_list = []
        memory_dict_list = []
        error_list = []
        # gonna be the list of things to execute, so that we can random shuffle it !
        execution_details_list = []

        memory_limit = data["memory_limit"]

        if memory_limit is None:
            # we get the memory allocated for code execution
            memory_limit = current_process.memory_info().text
            memory_limit = int(
                min(memory_limit * 1000 * 0.95, 2000000000 * 0.95))
        else:
            memory_limit = int(memory_limit)

        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

        # raise Exception(str((psutil.virtual_memory(), current_process.memory_info())))

        process_ids = str((
            cpu_id_list,
            psutil.cpu_count(logical=True),
            psutil.cpu_count(logical=False),
            current_process.pid,
            current_process.ppid(),
            current_process.cpu_affinity(),
            current_process.cpu_num()
        ))

        def handler(signum, frame):
            signame = signal.Signals(signum).name
            # print(f'Signal handler called with signal {signame} ({signum})')
            raise Exception("Couldn't open device!")

        # Set the signal handler and a 5-second alarm
        signal.signal(signal.SIGALRM, handler)

        template_list = [
            (
                Template('print(type($first$second$third))'),
                Template('$first$second$third')
            ),
            (
                Template('print(type(0b$first$second$third))'),
                Template('0b$first$second$third')
            ),
            (
                Template('print(type(\'$first$second$third\'))'),
                Template('\'$first$second$third\'')
            )
        ]

        replace_pair_list = [
            ('null', 'None'),
            ('<u>', ''),
            ('<u\>', ''),
            ('</u>', ''),
            ('false', 'False'),
            ('true', 'True')
        ]
        old_stdin = sys.stdin
        old_stdout = sys.stdout
        old_open = builtins.open

        faulthandler.enable()
        sys.stdin = StringIO()
        sys.stdout = StringIO()

        builtins.open = mock_open()

        signal.alarm(timeout)

        index_to_source_tuple_dict = dict()

        temp_file_name_counter = 0

        for index, question_inputs in enumerate(question_inputs_list):
            signal.alarm(giga_timeout)
            gc.collect()
            variable_name_to_input_dict = question_inputs['variable_name_to_input_dict']
            variable_json_list = []
            raise_issue = False
            temp_error_list = []
            runtime_dict = {}
            debugging_dict = {}
            memory_dict = {}

            runtime_dict_list.append(runtime_dict)
            debugging_dict_list.append(debugging_dict)
            memory_dict_list.append(memory_dict)
            error_list.append(temp_error_list)

            if input_handler == "with_dataclass":
                # in this case we need to first parse the input stream string into the dataclass to be able to run the code that follows :)
                dataclass_code = question_inputs["dataclass_code"]
                dataclass_code += (
                    "input_str = input()\n"
                    "input_str = input_str.encode().decode('unicode_escape')\n"
                    "input_cls = Input.from_str(str_escaped(input_str))\n"
                    "names = [x.name for x in input_cls.__dataclass_fields__.values()]\n"
                    "values = [getattr(input_cls, x) for x in names]\n"
                    "field_dict = dict(zip(names, values))\n"
                    "for x,y in field_dict.items():\n"
                    "    print(x)\n"
                    "    print(y)\n"
                )
                inputs = variable_name_to_input_dict.encode(
                    'unicode_escape').decode()

                try:
                    signal.alarm(timeout)
                    compiled = compile(dataclass_code, "<string>", "exec")
                    signal.alarm(timeout)
                except Exception as e:
                    signal.alarm(timeout)
                    continue

                if isinstance(inputs, list):
                    inputs = "\\n".join(inputs) + "\\n"

                try:
                    sys.stdin = StringIO(inputs)
                    sys.stdout = StringIO()

                    builtins.open = mock_open(read_data=inputs)
                    signal.alarm(timeout)
                    exec(compiled, {})
                    output = sys.stdout.getvalue()
                    signal.alarm(timeout)

                except Exception as e:
                    signal.alarm(timeout)
                    continue

                split_output = output.split('\n')
                assert split_output[-1] == ''
                split_output = split_output[:-1]
                assert len(split_output) % 2 == 0

                variable_name_to_input_dict = {}
                for i in range(len(split_output)//2):
                    variable_name_to_input_dict[split_output[2*i]
                                                ] = split_output[2*i+1]

                assert len(variable_name_to_input_dict) * \
                    2 == len(split_output)

                question_inputs_list[index]["variable_name_to_input_dict"] = variable_name_to_input_dict

            builtins.open = mock_open()

            #  We first parse the input types and get them ready
            for variable_name, input_ in variable_name_to_input_dict.items():
                assert type(input_) == str

                for replace_pair in replace_pair_list:
                    input_ = input_.replace(replace_pair[0], replace_pair[1])

                found_template = False
                for source_template, base_input_template in template_list:
                    temp_variable_json_list = []
                    source = source_template.safe_substitute(
                        first=input_, second='', third='')

                    try:
                        signal.alarm(timeout)
                        compiled = compile(source, "<source>", "exec")
                        linecache.cache["<source>"] = (
                            len(source),
                            None,
                            source.splitlines(True),
                            "<source>",
                        )
                        signal.alarm(timeout)
                    except Exception as e:
                        signal.alarm(timeout)
                        # temp_error_list.append(str(traceback.format_exc()))
                        # temp_error_list.append(str(e))
                        # output_w.send_bytes(
                        #     json.dumps({"error": str(e)}).encode("utf8")
                        # )
                        continue

                    # faulthandler.enable()

                    try:
                        sys.stdin = StringIO()
                        sys.stdout = StringIO()

                        # builtins.open = mock_open()
                        signal.alarm(timeout)
                        exec(compiled, {})
                        output = sys.stdout.getvalue()
                        signal.alarm(timeout)

                    except Exception as e:
                        signal.alarm(timeout)
                        continue

                    variable_json = {
                        'variable_name': variable_name,
                        'variable_type': output,
                        'base_input': base_input_template.safe_substitute(
                            first=input_,
                            second='',
                            third=''
                        ),
                    }

                    temp_variable_json_list.append(variable_json)

                    if output == "<class 'list'>\n":
                        source = source_template.safe_substitute(
                            first=input_, second='[0]', third='')

                        try:
                            signal.alarm(timeout)
                            compiled = compile(source, "<source>", "exec")
                            linecache.cache["<source>"] = (
                                len(source),
                                None,
                                source.splitlines(True),
                                "<source>",
                            )
                            signal.alarm(timeout)
                        except Exception as e:
                            signal.alarm(timeout)
                            continue

                        # faulthandler.enable()

                        try:
                            sys.stdin = StringIO()
                            sys.stdout = StringIO()

                            # builtins.open = mock_open()
                            signal.alarm(timeout)
                            exec(compiled, {})
                            output_2 = output + sys.stdout.getvalue()
                            signal.alarm(timeout)

                        except Exception as e:
                            signal.alarm(timeout)
                            continue

                        variable_json_2 = {
                            'variable_name': variable_name,
                            'variable_type': output_2,
                            'base_input': base_input_template.safe_substitute(
                                first=input_,
                                second='',
                                third=''
                            ),
                        }

                        temp_variable_json_list.append(variable_json_2)

                        if output_2 == "<class 'list'>\n<class 'list'>\n" or output_2 == "<class 'list'>\n<class 'tuple'>\n":
                            source = source_template.safe_substitute(
                                first=input_, second='[0]', third='[0]')

                            try:
                                signal.alarm(timeout)
                                compiled = compile(source, "<source>", "exec")
                                linecache.cache["<source>"] = (
                                    len(source),
                                    None,
                                    source.splitlines(True),
                                    "<source>",
                                )
                                signal.alarm(timeout)

                            except Exception as e:
                                signal.alarm(timeout)
                                continue

                            # faulthandler.enable()

                            try:
                                sys.stdin = StringIO()
                                sys.stdout = StringIO()

                                # builtins.open = mock_open()
                                signal.alarm(timeout)
                                exec(compiled, {})
                                output_3 = sys.stdout.getvalue()

                                if output_3 == "<class 'list'>\n" or output_3 == "<class 'tuple'>\n":
                                    raise Exception(
                                        'too many nested lists !!!!!!!!!!!!')

                                output_3 = output_2 + output_3
                                signal.alarm(timeout)

                            except Exception as e:
                                signal.alarm(timeout)
                                continue

                            variable_json_3 = {
                                'variable_name': variable_name,
                                'variable_type': output_3,
                                'base_input': base_input_template.safe_substitute(
                                    first=input_,
                                    second='',
                                    third=''
                                ),
                            }

                            temp_variable_json_list.append(variable_json_3)

                    found_template = True
                    if len(temp_variable_json_list) == 1:
                        # We are not handling a list
                        temp_ = temp_variable_json_list[0]
                        assert 'list' not in temp_['variable_type']
                        temp_['dimension'] = None
                        variable_json_list.append(temp_)

                    elif len(temp_variable_json_list) == 2:
                        temp_ = temp_variable_json_list[-1]
                        assert 'list' in temp_['variable_type']
                        temp_['dimension'] = 1

                        variable_json_list.append(temp_)

                    elif len(temp_variable_json_list) == 3:
                        temp_ = copy.deepcopy(temp_variable_json_list[-1])
                        assert 'list' in temp_['variable_type']
                        temp_['dimension'] = 1

                        variable_json_list.append(temp_)

                        if 'tuple' not in temp_['variable_type']:
                            temp_ = copy.deepcopy(temp_variable_json_list[-1])
                            assert 'list' in temp_['variable_type']
                            temp_['dimension'] = 2

                            variable_json_list.append(temp_)

                    else:
                        raise Exception(str(temp_variable_json_list))

                    break

                if not found_template:
                    raise_issue = True
                    break

            if raise_issue:
                if len(error_list[index]) == 0:
                    error_list[index].append(
                        ('problem with the parsing of the inputs' + str(variable_name_to_input_dict)))
                runtime_dict_list[index] = runtime_dict
                debugging_dict_list[index] = debugging_dict
                memory_dict_list[index] = memory_dict
                continue

            # First we need to get the executable code

            if input_handler != "with_dataclass":
                source = import_statements + wrap_code_to_get_executable_function(
                    question_inputs['code_content'],
                    list(question_inputs['variable_name_to_input_dict'].keys()),
                    input_handler,
                )

                try:
                    signal.alarm(timeout)
                    compiled = compile(source, "<source>", "exec")
                    linecache.cache["<source>"] = (
                        len(source),
                        None,
                        source.splitlines(True),
                        "<source>",
                    )
                    signal.alarm(timeout)
                except Exception as e:
                    signal.alarm(timeout)
                    if len(error_list[index]) == 0:
                        error_list[index].append((
                            # str(e),
                            # question_inputs['code_content'],
                            # output,
                            # list(question_inputs['variable_name_to_input_dict'].keys()),
                            'cannot get the function to execute' +
                            '####' + str(e) + "hey1"
                        )[:3000])
                    runtime_dict_list[index] = runtime_dict
                    debugging_dict_list[index] = debugging_dict
                    memory_dict_list[index] = memory_dict
                    continue

                # faulthandler.enable()

                try:
                    sys.stdin = StringIO()
                    sys.stdout = StringIO()

                    # builtins.open = mock_open()
                    signal.alarm(timeout)
                    exec(compiled, {})
                    output = sys.stdout.getvalue()

                    assert "error" not in output.lower()

                    if output.split('\n')[0] == 'None':
                        raise Exception(
                            'yolo' + str(variable_name_to_input_dict))

                    function_to_execute = output.split('\n')[0].split(' ')[1]
                    variable_name_to_argument_name = output.split('\n')[1]
                    class_to_self_call = output.split('\n')[2]
                    signal.alarm(timeout)

                except Exception as e:
                    signal.alarm(timeout)

                    if len(error_list[index]) == 0:
                        error_list[index].append((
                            'cannot get the function to execute' +
                            '####' + str(e) + 'hey2'
                        )[:3000])
                    runtime_dict_list[index] = runtime_dict
                    debugging_dict_list[index] = debugging_dict
                    memory_dict_list[index] = memory_dict
                    continue

                class_to_self_call = str(class_to_self_call)
                if class_to_self_call != 'None':
                    class_to_self_call = class_to_self_call.split('\'')[1]

                    if 'ContextWrapperForTimeSpaceComplexity' not in class_to_self_call:
                        class_to_self_call = 'ContextWrapperForTimeSpaceComplexity.' + class_to_self_call

                # HANDLE ROOT FUNCTIONS
                source = import_statements + \
                    make_root_functions_accessible_inside_classes(
                        question_inputs['code_content'])

                try:
                    signal.alarm(timeout)
                    compiled = compile(source, "<source>", "exec")
                    linecache.cache["<source>"] = (
                        len(source),
                        None,
                        source.splitlines(True),
                        "<source>",
                    )
                    signal.alarm(timeout)
                except Exception as e:
                    signal.alarm(timeout)
                    if len(error_list[index]) == 0:
                        error_list[index].append(
                            str('compilation problem with root functions accessible', source)[:3000])
                    runtime_dict_list[index] = runtime_dict
                    debugging_dict_list[index] = debugging_dict
                    memory_dict_list[index] = memory_dict
                    continue

                # faulthandler.enable()

                try:
                    sys.stdin = StringIO()
                    sys.stdout = StringIO()

                    # deal with the case of open(0).read()
                    # builtins.open = mock_open()
                    # builtins.open = mock_open(read_data=input_)
                    signal.alarm(timeout)
                    exec(compiled, {})
                    functions_to_change = sys.stdout.getvalue()
                    signal.alarm(timeout)

                except Exception as e:
                    signal.alarm(timeout)
                    if len(error_list[index]) == 0:
                        error_list[index].append(
                            str('execution problem with root functions accessible', source)[:3000])
                    runtime_dict_list[index] = runtime_dict
                    debugging_dict_list[index] = debugging_dict
                    memory_dict_list[index] = memory_dict
                    continue

                # we change the code input
                code_to_execute = add_context_to_root_functions(
                    question_inputs['code_content'],
                    functions_to_change
                )

                # HANDLE ROOT OBJECTS
                source = import_statements + \
                    make_root_objects_accessible_inside_classes(
                        question_inputs['code_content'])

                try:
                    signal.alarm(timeout)
                    compiled = compile(source, "<source>", "exec")
                    linecache.cache["<source>"] = (
                        len(source),
                        None,
                        source.splitlines(True),
                        "<source>",
                    )
                    signal.alarm(timeout)
                except Exception as e:
                    signal.alarm(timeout)
                    if len(error_list[index]) == 0:
                        error_list[index].append(
                            str('compilation problem with root objects accessible', source)[:3000])
                    runtime_dict_list[index] = runtime_dict
                    debugging_dict_list[index] = debugging_dict
                    memory_dict_list[index] = memory_dict
                    continue

                # faulthandler.enable()

                try:
                    sys.stdin = StringIO()
                    sys.stdout = StringIO()

                    # deal with the case of open(0).read()
                    # builtins.open = mock_open()
                    # builtins.open = mock_open(read_data=input_)
                    signal.alarm(timeout)
                    exec(compiled, {})
                    objects_to_change = sys.stdout.getvalue()
                    signal.alarm(timeout)

                except Exception as e:
                    signal.alarm(timeout)
                    if len(error_list[index]) == 0:
                        error_list[index].append(
                            str('execution problem with root objects accessible', source)[:3000])
                    runtime_dict_list[index] = runtime_dict
                    debugging_dict_list[index] = debugging_dict
                    memory_dict_list[index] = memory_dict
                    continue

                # we change the code input
                code_to_execute = add_context_to_root_objects(
                    code_to_execute,
                    objects_to_change
                )

            elif input_handler == 'with_dataclass':
                code_to_execute = question_inputs['code_content']

            else:
                raise Exception('input handler not handled')

            if correct_nlogn:
                code_to_execute = replace_sorting_algorithm(
                    code_to_execute
                )

            assert set(question_inputs['variable_name_to_input_dict'].keys()) == set(
                map(lambda x: x['variable_name'], variable_json_list))

            variable_name_type_dimension_to_base_input_dict_dict = {}

            #  This part pre-compute some inputs that are going to be used a lot at code execution time to save time
            try:
                for variable_json in variable_json_list:
                    variable_type = variable_json['variable_type']
                    base_input = variable_json['base_input']
                    variable_name = variable_json['variable_name']
                    dimension = variable_json['dimension']

                    if (variable_type, dimension) not in variable_type_dimension_method_to_base_multiplier_priority.keys():
                        raise Exception(
                            'not handled' + variable_type + str(dimension))

                    if not any(list(map(
                        lambda x: x['used_for_base_input_precomputation'],
                        list(variable_type_dimension_method_to_base_multiplier_priority[
                            (variable_type, dimension)
                        ].values())
                    ))):
                        # for this type, no method is defined for the base input precomputation
                        raise Exception(
                            'no used_for_base_input_precomputation' + variable_type + str(dimension))

                    variable_name_type_dimension_to_base_input_dict_dict[(
                        variable_name, variable_type, dimension)] = {}

                    for method in variable_type_dimension_method_to_base_multiplier_priority[
                        (variable_type, dimension)
                    ].keys():
                        signal.alarm(giga_timeout)

                        local_variable_name_set = set(locals().keys())

                        for local_variable_name in [
                            "data",
                            "source",
                            "compiled",
                            "output",
                            "output_1",
                            "output_2",
                            "varied_input",
                            "source_1",
                            "source_2",
                            "error_that_was_raised",
                            "_",
                        ]:
                            if local_variable_name in local_variable_name_set:
                                locals()[local_variable_name] = 0
                                del locals()[local_variable_name]

                        gc.collect()

                        base = variable_type_dimension_method_to_base_multiplier_priority[
                            (variable_type, dimension)
                        ][method]['base']
                        multiplier = variable_type_dimension_method_to_base_multiplier_priority[
                            (variable_type, dimension)
                        ][method]['multiplier']
                        used_for_base_input_precomputation = variable_type_dimension_method_to_base_multiplier_priority[
                            (variable_type, dimension)
                        ][method]['used_for_base_input_precomputation']

                        if not used_for_base_input_precomputation:
                            continue

                        source = import_statements + base(base_input)

                        try:
                            signal.alarm(timeout)
                            compiled = compile(source, "<source>", "exec")
                            linecache.cache["<source>"] = (
                                len(source),
                                None,
                                source.splitlines(True),
                                "<source>",
                            )
                            signal.alarm(timeout)
                        except Exception as e:
                            signal.alarm(timeout)
                            raise Exception(source)

                        # faulthandler.enable()
                        try:
                            sys.stdin = StringIO()
                            sys.stdout = StringIO()
                            # deal with the case of open(0).read()
                            # builtins.open = mock_open()
                            # builtins.open = mock_open(read_data=input_)
                            signal.alarm(timeout)
                            exec(compiled, {})
                            output = sys.stdout.getvalue().splitlines()[0]
                            signal.alarm(timeout)
                        except Exception as e:
                            signal.alarm(timeout)
                            raise Exception('error when multiplying for SMALL')

                        varied_input = output

                        if variable_type == "<class 'str'>\n":
                            varied_input = '"' + varied_input + '"'

                        variable_name_type_dimension_to_base_input_dict_dict[
                            (variable_name, variable_type, dimension)
                        ][(method, 'small')] = varied_input

                        #########
                        source = import_statements + multiplier(
                            varied_input, size_of_other_arguments
                        )

                        try:
                            signal.alarm(timeout)
                            compiled = compile(source, "<source>", "exec")
                            linecache.cache["<source>"] = (
                                len(source),
                                None,
                                source.splitlines(True),
                                "<source>",
                            )
                            signal.alarm(timeout)
                            # faulthandler.enable()

                            sys.stdin = StringIO()
                            sys.stdout = StringIO()

                            # deal with the case of open(0).read()
                            # builtins.open = mock_open()
                            # builtins.open = mock_open(read_data=input_)
                            signal.alarm(giga_timeout)
                            exec(compiled, {})
                            output = sys.stdout.getvalue().splitlines()[0]
                            signal.alarm(timeout)

                        except Exception as e:
                            #########
                            source = import_statements + multiplier(
                                varied_input, max(
                                    size_of_other_arguments//10, 3)
                            )

                            try:
                                signal.alarm(timeout)
                                compiled = compile(source, "<source>", "exec")
                                linecache.cache["<source>"] = (
                                    len(source),
                                    None,
                                    source.splitlines(True),
                                    "<source>",
                                )
                                signal.alarm(timeout)
                                # faulthandler.enable()

                                sys.stdin = StringIO()
                                sys.stdout = StringIO()

                                # deal with the case of open(0).read()
                                # builtins.open = mock_open()
                                # builtins.open = mock_open(read_data=input_)
                                signal.alarm(giga_timeout)
                                exec(compiled, {})
                                output = sys.stdout.getvalue().splitlines()[0]
                                signal.alarm(timeout)

                            except Exception as e:
                                signal.alarm(timeout)
                                #  Not able to generate this larger input
                                raise Exception(
                                    'error when multiplying for LARGE')
                            #########
                        #########

                        varied_input = output

                        # to be refactored (move to generate_inputs.py)
                        if variable_type == "<class 'str'>\n":
                            varied_input = '"' + varied_input + '"'

                        variable_name_type_dimension_to_base_input_dict_dict[
                            (variable_name, variable_type, dimension)
                        ][(method, 'large')] = varied_input

            except Exception as e:
                signal.alarm(timeout)
                if len(error_list[index]) == 0:
                    error_list[index].append(str('variable_name_type_dimension_to_base_input_dict_dict', str(
                        e), str(traceback.format_exc()))[:3000])
                runtime_dict_list[index] = runtime_dict
                debugging_dict_list[index] = debugging_dict
                memory_dict_list[index] = memory_dict
                continue

            # This part registers runs to execute, with all the code+input part ready
            signal.alarm(timeout)

            if input_handler != "with_dataclass":
                source_1, source_2 = get_source_tuple(
                    code_content=code_to_execute,
                    function_to_execute=str(function_to_execute),
                    variable_name_to_argument_name=str(
                        variable_name_to_argument_name),
                    class_to_self_call=str(class_to_self_call),
                    embed_cprofile=time_profiler,
                )
            elif input_handler == "with_dataclass":
                source_1, source_2 = get_source_tuple_with_dataclass(
                    code_content=code_to_execute,
                    dataclass_code=question_inputs['dataclass_code'],
                    inputs=inputs,  # defined above but only for this input_handler
                    embed_cprofile=time_profiler,
                )
            else:
                raise Exception('not handled')

            source_1 = import_statements + source_1

            index_to_source_tuple_dict[index] = (source_1, source_2)

            try:
                for variable_json in variable_json_list:
                    # let's evaluate input by input
                    signal.alarm(timeout)
                    variable_type_ref = variable_json['variable_type']
                    variable_name_ref = variable_json['variable_name']
                    dimension_ref = variable_json['dimension']
                    # base_input_ref = variable_json['base_input']

                    if (variable_type_ref, dimension_ref) not in variable_type_dimension_method_to_base_multiplier_priority.keys():
                        raise Exception(
                            'not handled' + variable_type_ref + dimension_ref)

                    runtime_dict[str_variable_name_type_dimension(
                        variable_name_ref, variable_type_ref, dimension_ref)] = dict()
                    debugging_dict[str_variable_name_type_dimension(
                        variable_name_ref, variable_type_ref, dimension_ref)] = dict()
                    memory_dict[str_variable_name_type_dimension(
                        variable_name_ref, variable_type_ref, dimension_ref)] = dict()

                    multiplier_current = multiplier_start

                    while multiplier_current <= multiplier_end:
                        runtime_dict[str_variable_name_type_dimension(
                            variable_name_ref, variable_type_ref, dimension_ref)][multiplier_current] = []
                        debugging_dict[str_variable_name_type_dimension(
                            variable_name_ref, variable_type_ref, dimension_ref)][multiplier_current] = []
                        memory_dict[str_variable_name_type_dimension(
                            variable_name_ref, variable_type_ref, dimension_ref)][multiplier_current] = []

                        for expansion_details in generate_expansion_details_list(
                            variable_type_dimension_to_expansion_details_list,
                            variable_name_type_dimension_to_base_input_dict_dict,
                            variable_name_ref,
                            variable_type_ref,
                            dimension_ref,
                            # do_all_cases,
                            # have_other_arguments_large,
                        ):
                            signal.alarm(giga_timeout)

                            local_variable_name_set = set(locals().keys())

                            for local_variable_name in [
                                "data",
                                "source",
                                "compiled",
                                "output",
                                "output_1",
                                "output_2",
                                "varied_input",
                                "source_1",
                                "source_2",
                                "error_that_was_raised",
                                "_",
                            ]:
                                if local_variable_name in local_variable_name_set:
                                    locals()[local_variable_name] = 0
                                    del locals()[local_variable_name]

                            gc.collect()

                            # for _ in range(multiplier_repeat):
                            signal.alarm(timeout)
                            variable_name_to_input_dict = copy.deepcopy(
                                expansion_details['variable_name_to_input_dict'])
                            variable_name_to_multiplier_dict = expansion_details[
                                'variable_name_to_multiplier_dict']
                            tag_list = expansion_details['tag_list']
                            id_ = expansion_details['id_']
                            priority = expansion_details['priority']

                            error_with_current_expansion = False
                            error_that_was_raised = None

                            for variable_name, multiplier in variable_name_to_multiplier_dict.items():
                                if multiplier is None:
                                    continue

                                varied_input = multiplier(
                                    variable_name_to_input_dict[variable_name], multiplier_current
                                )

                                #########
                                source = import_statements + varied_input

                                try:
                                    signal.alarm(timeout)
                                    compiled = compile(
                                        source, "<source>", "exec")
                                    linecache.cache["<source>"] = (
                                        len(source),
                                        None,
                                        source.splitlines(True),
                                        "<source>",
                                    )
                                    signal.alarm(timeout)

                                except Exception as e:
                                    signal.alarm(timeout)
                                    error_with_current_expansion = True
                                    error_that_was_raised = str(
                                        e) + str(str(traceback.format_exc()))
                                    break

                                # faulthandler.enable()

                                try:
                                    sys.stdin = StringIO()
                                    sys.stdout = StringIO()

                                    # deal with the case of open(0).read()
                                    # builtins.open = mock_open()
                                    # builtins.open = mock_open(read_data=input_)
                                    signal.alarm(large_timeout)
                                    exec(compiled, {})
                                    output = sys.stdout.getvalue().splitlines()[
                                        0]
                                    signal.alarm(timeout)

                                except Exception as e:
                                    signal.alarm(timeout)
                                    error_with_current_expansion = True
                                    error_that_was_raised = str(
                                        e) + str(str(traceback.format_exc()))
                                    break
                                #########

                                varied_input = output

                                variable_type = list(
                                    filter(lambda x: x[0] == variable_name, list(
                                        variable_name_type_dimension_to_base_input_dict_dict.keys()))
                                )[0][1]

                                if variable_type == "<class 'str'>\n":
                                    varied_input = '"' + varied_input + '"'

                                variable_name_to_input_dict[variable_name] = varied_input

                            # certain classes need a postprocessing
                            # should be refactored and coming from generate_inputs.py
                            if error_with_current_expansion:
                                if len(error_list[index]) == 0:
                                    error_list[index].append(('multiplier input compilation error' + '####' + str(
                                        (len(execution_details_list),
                                         id_, variable_name_to_input_dict)
                                    ) + "#######" + str(error_that_was_raised))[:3000])
                                continue

                            for variable_name, varied_input in variable_name_to_input_dict.items():
                                variable_type = list(
                                    filter(lambda x: x[0] == variable_name, list(
                                        variable_name_type_dimension_to_base_input_dict_dict.keys()))
                                )[0][1]

                                variable_name_to_input_dict[variable_name] = varied_input

                            signal.alarm(timeout)

                            variable_name_to_input_dict_stringified = get_variable_name_to_input_dict_stringified(
                                variable_name_to_input_dict=variable_name_to_input_dict,
                            )

                            temp_file_name = temp_file_name_seed + \
                                '_{}.txt'.format(temp_file_name_counter)
                            temp_file_name_counter += 1

                            signal.alarm(giga_timeout)

                            builtins.open = old_open
                            with open(temp_file_name, 'w') as f:
                                f.write(variable_name_to_input_dict_stringified)
                            builtins.open = mock_open()

                            for _ in range(multiplier_repeat):
                                execution_details_list.append(
                                    {
                                        'temp_file_name': temp_file_name,
                                        'variable_name': variable_name_ref,
                                        'variable_type': variable_type_ref,
                                        'dimension': dimension_ref,
                                        'multiplier_current': multiplier_current,
                                        'id_': id_,
                                        'tag_list': tag_list,
                                        'index': index,
                                        'function_to_execute': function_to_execute if input_handler != "with_dataclass" else "function_wrapper_for_time_space_complexity",
                                        'priority': priority,
                                    }
                                )

                            del variable_name_to_input_dict
                            del variable_name_to_input_dict_stringified
                            signal.alarm(giga_timeout)
                            gc.collect()

                            runtime_dict_list[index][
                                str_variable_name_type_dimension(
                                    variable_name_ref, variable_type_ref, dimension_ref)
                            ][multiplier_current].append(
                                {
                                    'value_list': [],
                                    'id_': id_,
                                    'tag_list': tag_list,
                                    'priority': priority,
                                }
                            )

                            memory_dict_list[index][
                                str_variable_name_type_dimension(
                                    variable_name_ref, variable_type_ref, dimension_ref)
                            ][multiplier_current].append(
                                {
                                    'value_list': [],
                                    'id_': id_,
                                    'tag_list': tag_list,
                                    'priority': priority,
                                }
                            )

                        multiplier_current = min(
                            multiplier_op(multiplier_current,
                                          multiplier_mult_step),
                            multiplier_max_increase + multiplier_current,
                        )

            except Exception as e:
                signal.alarm(timeout)
                if len(error_list[index]) == 0:
                    error_list[index].append(str('prepation execution details list', str(
                        e), str(traceback.format_exc()))[:3000])
                runtime_dict_list[index] = runtime_dict
                debugging_dict_list[index] = debugging_dict
                memory_dict_list[index] = memory_dict
                continue

        if shuffle_runs:
            random.shuffle(execution_details_list)

        error_evaluating = False

        ignore_excution_dict = collections.defaultdict(lambda: False)

        # key_value_memory_list = list(locals().items())
        # key_value_memory_list_to_print = []

        # for key_, value_ in key_value_memory_list:
        #     if '_' not in key_:
        #         key_value_memory_list_to_print.append((key_, sys.getsizeof(value_)))

        # raise Exception(str(key_value_memory_list_to_print))

        # counter_locals = 0

        for execution_details in execution_details_list:

            # if counter_locals <= 1:
            # error_list[index].append((str(("###counter_locals", counter_locals, str(locals().keys())))))
            # counter_locals += 1

            # now we need to do some manual memory cleanup !
            signal.alarm(giga_timeout)

            local_variable_name_set = set(locals().keys())

            for local_variable_name in [
                "data",
                "question_inputs_list",
                "variable_type_dimension_method_to_base_multiplier_priority",
                "variable_type_dimension_to_expansion_details_list",
                "question_inputs",
                "variable_json_list",
                "runtime_dict",
                "debugging_dict",
                "memory_dict",
                "temp_error_list",
                "source",
                "compiled",
                "output",
                "output_2",
                "output_1",
                "code_to_execute",
                "variable_name_type_dimension_to_base_input_dict_dict",
                "varied_input",
                "source_1",
                "source_2",
                "error_that_was_raised",
                "f",
                "_",
                "variable_name_to_input_dict_stringified",
                "temp_runtime",
                "temp_memory",
                "runtime_details",
                "memory_details",
            ]:
                if local_variable_name in local_variable_name_set:
                    locals()[local_variable_name] = 0
                    del locals()[local_variable_name]

            gc.collect()

            local_variable_name_set = set(locals().keys())
            for local_variable_name in local_variable_name_set:
                if local_variable_name not in [
                    "input_r",
                    "output_w",
                    "print_debugging_dict_list",
                    "cpu_id_list",
                    "cpu_id",
                    "time_profiler",
                    "timeout",
                    "large_timeout",
                    "giga_timeout",
                    "import_statements",
                    "runtime_dict_list",
                    "debugging_dict_list",
                    "memory_dict_list",
                    "error_list",
                    "execution_details",
                    "execution_details_list",
                    "pid",
                    "current_process",
                    "memory_limit",
                    "process_ids",
                    "handler",
                    "old_open",
                    "index_to_source_tuple_dict",
                    "error_evaluating",
                    "ignore_excution_dict",
                ]:
                    locals()[local_variable_name] = 0
                    del locals()[local_variable_name]

            gc.collect()

            temp_file_name = execution_details['temp_file_name']
            builtins.open = old_open
            with open(temp_file_name, 'r') as f:
                variable_name_to_input_dict_stringified = f.read()
            builtins.open = mock_open()

            variable_name = execution_details['variable_name']
            variable_type = execution_details['variable_type']
            dimension = execution_details['dimension']
            multiplier_current = execution_details['multiplier_current']
            id_ = execution_details['id_']
            tag_list = execution_details['tag_list']
            index = execution_details['index']
            function_to_execute = execution_details['function_to_execute']
            priority = execution_details['priority']

            source = index_to_source_tuple_dict[index][0] + \
                variable_name_to_input_dict_stringified + \
                index_to_source_tuple_dict[index][1]

            if ignore_excution_dict[(
                variable_name,
                variable_type,
                dimension,
                multiplier_current,
                id_,
                index,
                priority,
            )]:
                continue

            source_to_benchmark = import_statements + \
                "\nclass ClassWrapper:\n    def benchmark_function():\n        for i in range(10000):   \n            i * i * i * i * i\n        return\n    \nfunction_to_execute = ClassWrapper.benchmark_function\ntracemalloc.start()\ntracemalloc.clear_traces()\ntracemalloc.reset_peak()\n\ncpr = cProfile.Profile()\ncpr.enable()\n\ntry:\n    output_values = function_to_execute()\n\n    current_memory, peak_memory = tracemalloc.get_traced_memory()\n    del output_values\n    after_memory, _ = tracemalloc.get_traced_memory()\n    print(peak_memory - current_memory + after_memory)\n\nfinally:\n    cpr.disable()\n    tracemalloc.stop()\n\nfunction_to_execute_name = str(function_to_execute).split('.')[-1].split(' ')[0]\nfound = False\nfor entry in cpr.getstats():\n    if type(entry.code) != str:\n        function_name = entry.code.co_name\n\n        if function_name == function_to_execute_name:\n            found = True\n            print(entry.totaltime)\n            break\n\nif not found:\n    raise Exception('Could not get the execution time')\n"

            try:
                signal.alarm(giga_timeout)
                linecache.clearcache()
                compiled = compile(
                    source,
                    "<source>",
                    "exec",
                    flags=0,
                    dont_inherit=False,
                    optimize=0,
                )
                linecache.cache["<source>"] = (
                    len(source),
                    None,
                    source.splitlines(True),
                    "<source>",
                )
                signal.alarm(timeout)

                signal.alarm(giga_timeout)
                compiled_to_benchmark = compile(
                    source_to_benchmark,
                    "<source_to_benchmark>",
                    "exec",
                    flags=0,
                    dont_inherit=False,
                    optimize=0,
                )
                signal.alarm(timeout)
            except Exception as e:
                signal.alarm(timeout)
                ignore_excution_dict[(
                    variable_name,
                    variable_type,
                    dimension,
                    multiplier_current,
                    id_,
                    index,
                    priority,
                )] = True
                if len(error_list[index]) == 0:
                    error_list[index].append(
                        ('multiplier code compilation error' + '####' + str(e))[:3000])
                continue

            # faulthandler.enable()

            try:
                if print_debugging_dict_list:
                    # sys.stdin = old_stdin
                    # sys.stdout = old_stdout
                    builtins.open = old_open
                    cpu_utilisation_before = (
                        current_process.cpu_percent(),
                        current_process.cpu_num(),
                        psutil.virtual_memory().percent,
                        current_process.memory_info()[0]/2.**30
                    )
                    signal.alarm(large_timeout)
                    builtins.open = mock_open()
                    # cpu_utilisation = 1

                # deal with the case of open(0).read()
                # builtins.open = mock_open()
                # builtins.open = mock_open(read_data=input_)
                # temp_runtime_list = []
                # temp_memory_list = []

                temp_runtime = None
                temp_memory = None

                temp_runtime_benchmark = None

                # for _ in range(multiplier_repeat):
                signal.alarm(large_timeout)
                sys.stdin = StringIO()
                sys.stdout = StringIO()

                if time_profiler == "time":
                    start_time = time.time()
                    try:
                        exec(compiled, {})
                    except Exception as e:
                        signal.alarm(timeout)
                        raise Exception('execution problem' + str(e))
                    end_time = time.time()
                    output = sys.stdout.getvalue().splitlines()[0]
                    temp_runtime = end_time - start_time
                    temp_memory = int(output)

                elif time_profiler == "timeit":
                    start_time = time.process_time()
                    try:
                        exec(compiled, {})
                    except Exception as e:
                        signal.alarm(timeout)
                        raise Exception('execution problem' + str(e))
                    end_time = time.process_time()
                    output = sys.stdout.getvalue().splitlines()[0]
                    temp_runtime = end_time - start_time
                    temp_memory = int(output)

                elif time_profiler == "cprofile":
                    cpr = cProfile.Profile()
                    cpr.enable()
                    try:
                        exec(compiled, {})
                    except Exception as e:
                        signal.alarm(timeout)
                        cpr.disable()
                        raise Exception('execution problem' + str(e))
                    cpr.disable()
                    output = sys.stdout.getvalue().splitlines()[0]
                    temp_memory = int(output)
                    function_to_execute_name = str(
                        function_to_execute).split('.')[-1].split(' ')[0]
                    found = False
                    for entry in cpr.getstats():
                        if type(entry.code) != str:
                            function_name = entry.code.co_name

                            if function_name == function_to_execute_name:
                                found = True
                                temp_runtime = entry.totaltime
                                break

                    if not found:
                        raise Exception('Could not get the execution time')

                # elif time_profiler == "cprofilewithin":
                #     try:
                #         exec(compiled, {})
                #     except Exception as e:
                #         signal.alarm(timeout)
                #         raise Exception('execution problem' + str(e))
                #     output_1, output_2 = sys.stdout.getvalue().splitlines()[:2]
                #     temp_memory = int(output_1)
                #     temp_runtime = float(output_2)

                elif time_profiler == "cprofilewithin":
                    try:
                        benchmark_value_list = []
                        for _ in range(10):
                            signal.alarm(large_timeout)
                            sys.stdin = StringIO()
                            sys.stdout = StringIO()
                            exec(compiled_to_benchmark, {}, {})

                            _, output_2 = sys.stdout.getvalue().splitlines()[
                                :2]
                            benchmark_value_list.append(float(output_2))

                        temp_runtime_benchmark_1 = np.median(
                            benchmark_value_list)

                        signal.alarm(large_timeout)
                        sys.stdin = StringIO()
                        sys.stdout = StringIO()
                        exec(compiled, {})
                        output_1, output_2 = sys.stdout.getvalue().splitlines()[
                            :2]
                        temp_memory = int(output_1)
                        temp_runtime = float(output_2)

                        benchmark_value_list = []
                        for _ in range(10):
                            signal.alarm(large_timeout)
                            sys.stdin = StringIO()
                            sys.stdout = StringIO()
                            exec(compiled_to_benchmark, {}, {})

                            _, output_2 = sys.stdout.getvalue().splitlines()[
                                :2]
                            benchmark_value_list.append(float(output_2))

                        temp_runtime_benchmark_2 = np.median(
                            benchmark_value_list)

                        temp_runtime = temp_runtime / \
                            (temp_runtime_benchmark_1 * temp_runtime_benchmark_2)

                    except Exception as e:
                        signal.alarm(timeout)
                        raise Exception('execution problem' + str(e))

                elif time_profiler == "cprofilerobust":
                    sw_interval = sys.getswitchinterval()
                    try:
                        sys.setswitchinterval(1000)
                        exec(compiled, {})
                    except Exception as e:
                        signal.alarm(timeout)
                        sys.setswitchinterval(sw_interval)
                        raise Exception('execution problem' + str(e))
                    sys.setswitchinterval(sw_interval)
                    output_1, output_2 = sys.stdout.getvalue().splitlines()[:2]
                    temp_memory = float(output_1)
                    temp_runtime = float(output_2)

                elif time_profiler == "cprofilearound":
                    sw_interval = sys.getswitchinterval()
                    try:
                        sys.setswitchinterval(1000)
                        exec(compiled, {})
                    except Exception as e:
                        signal.alarm(timeout)
                        sys.setswitchinterval(sw_interval)
                        raise Exception('execution problem' + str(e))
                    sys.setswitchinterval(sw_interval)
                    output_1, output_2 = sys.stdout.getvalue(
                    ).splitlines()[-2:]
                    temp_memory = int(output_1)
                    temp_runtime = float(output_2)

                else:
                    raise Exception('time profiler not handled')

                # output_list.append(output)

                for index_2, runtime_details in enumerate(
                    runtime_dict_list[index][str_variable_name_type_dimension(
                        variable_name, variable_type, dimension)][multiplier_current]
                ):
                    if runtime_details['id_'] == id_:
                        runtime_dict_list[index][
                            str_variable_name_type_dimension(
                                variable_name, variable_type, dimension)
                        ][multiplier_current][index_2]['value_list'].append(temp_runtime)

                for index_2, memory_details in enumerate(
                    memory_dict_list[index][str_variable_name_type_dimension(
                        variable_name, variable_type, dimension)][multiplier_current]
                ):
                    if memory_details['id_'] == id_:
                        memory_dict_list[index][
                            str_variable_name_type_dimension(
                                variable_name, variable_type, dimension)
                        ][multiplier_current][index_2]['value_list'].append(temp_memory)

                signal.alarm(timeout)
                if print_debugging_dict_list:
                    # sys.stdin = old_stdin
                    # sys.stdout = old_stdout
                    builtins.open = old_open
                    cpu_utilisation_after = (
                        current_process.cpu_percent(),
                        current_process.cpu_num(),
                        psutil.virtual_memory().percent,
                        current_process.memory_info()[0]/2.**30
                    )
                    signal.alarm(large_timeout)
                    builtins.open = mock_open()
                    debugging_dict[index][str_variable_name_type_dimension(variable_name, variable_type, dimension)][multiplier_current].append(
                        (cpu_utilisation_before, cpu_utilisation_after)
                    )

            except Exception as e:
                signal.alarm(timeout)
                ignore_excution_dict[(
                    variable_name,
                    variable_type,
                    dimension,
                    multiplier_current,
                    id_,
                    index,
                    priority,
                )] = True
                if len(error_list[index]) == 0 or not error_evaluating:
                    error_list[index].append((
                        'multiplier code execution error' +
                        '####' +
                        variable_name +
                        id_ +
                        '####' +
                        variable_name_to_input_dict_stringified[:100] +
                        '#####' +
                        function_to_execute +
                        '####' +
                        str(e) +
                        str(traceback.format_exc()) +
                        source
                    )[:3000])
                    error_evaluating = True
                continue

        signal.alarm(timeout)

        if cpu_id_list is not None:
            assert current_process.cpu_affinity() == cpu_id_list

        output_w.send_bytes(
            json.dumps(
                {
                    "runtime_dict_list": runtime_dict_list,
                    "debugging_dict_list": debugging_dict_list,
                    "memory_dict_list": memory_dict_list,
                    "error_list": error_list,
                    "process_ids": process_ids,
                }
            ).encode("utf8")
        )
    except Exception as e:
        output_w.send_bytes(
            json.dumps(
                {
                    "error": str(e) + str(traceback.format_exc()),
                }
            ).encode("utf8")
        )
    return


if __name__ == "__main__":
    main()
