# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import re
import random
import pytest
import psutil
import pickle
from pathlib import Path
import os
import numpy as np
import math
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import datetime
from typing import Any
from execution_measures.worker_launcher import exec_python_time_space_complexities
from utils import (
    correct_complexity_formatting,
    convert_measures_dict_format,
    convert_measures_set_id_to_input_properties_format,
)
from curve_fitting.fitting_utils import (
    map_true_complexity, equality_complexities
)
from curve_fitting.fitting_curve import (
    infer_complexity_from_values
)
from curve_fitting.fitting_class import (
    Constant,
    Linear,
    Quadratic,
    Cubic,
    Logarithmic,
    Linearithmic,
    Constant_2,
)
import sys


def run_complexity_framework(
    path_to_jsonl_file: str,
    sub_key: str = None,
    code_start_index: int = 0,
    code_end_index: int = None,
    filter_on_problem: str = None,
    multiply_samples_factor: int = 1,
    input_handler: str = "with_dataclass",
    log_outputs: bool = False,
    save_results: bool = True,
    skip_saving_full_results: bool = True,
    results_folder_name_root: str = 'results',
    shuffle_runs: bool = True,
    correct_nlogn: bool = True,
    multiplier_op: str = "multiplication",
    multiplier_start: int = 1,
    multiplier_repeat: int = 10,
    multiplier_end: int = 3100,
    multiplier_mult_step: int = 2,
    multiplier_max_increase: int = 512,
    size_of_other_arguments: int = 1000,
    time_profiler: str = "cprofilearound",
    filter_outliers: bool = False,
    apply_penalty: bool = True,
    apply_constraints: bool = True,
    zero_out_first_value: bool = True,
    piecewise_fit: bool = True,
    aggregate_y_values: str = 'most_stable_aggregate_aggregate',
    max_time_rate: float = 0.8,
    elect_complexity_time: str = 'min',
    elect_complexity_space: str = 'max',
    fix_constant_complexity: bool = False,
    fix_negligeable_complexity: bool = True,
    temp_file_name_seed: str = 'complexity',
    memory_limit: int = None,
    timeout: int = 1,
    large_timeout: int = 1,
    giga_timeout: int = 30,
    global_timeout: int = 1200,
    max_workers: int = 1000000,
    number_solutions_per_worker: int = 1,
    main_process_cpu_id_list: str = '0,1',
    forkserver_type: str = 'standard',
    use_distinct_forkservers: bool = False,
    forkserver_cpu_id_list: str = '2,3',
    sandbox_cpu_id_step: int = 4,
    sandbox_incremental_cpu_id_list: str = '0,1,2,3',
    distinct_forkservers_incremental_cpu_id_list: str = '0,1,2,3',
    slurm_array_task_id: int = None,
    slurm_array_task_max: int = None,
    slurm_array_task_min: int = None,
    slurm_array_job_id: int = None,
) -> None:
    """
    Run the complexity framework on a dataset of Python code.

    This function takes in a dataset of Python code, runs the complexity framework on it,
    and returns the results.

    The dataset should be in a JSONL file, where each line represents a sample of code.
    Each sample should have the following fields:

    * problem_name: str
    * problem_id: str
    * solution_id: str
    * solution_code: str
    * inputs_example: str

    If the input_handler is set to "with_dataclass", then each sample should also have a
    dataclass_code field.

    * dataclass_code: str

    The function can be configured to run on a subset of the dataset by specifying the
    code_start_index and code_end_index parameters. It can also be configured to filter
    on a specific problem name by specifying the filter_on_problem parameter.

    The function can be run multiple times on each sample by specifying the
    multiply_samples_factor parameter.

    The results of the complexity framework can be saved to a folder by specifying the
    save_results parameter. The name of the folder can be specified using the
    results_folder_name_root parameter.

    Various other parameters can be used to configure the complexity framework, such as
    the time profiler to use, whether to apply penalties to the different complexity
    classes, and whether to use constraints during curve fitting.

    Parameters:



    *** DATASET ***

    path_to_jsonl_file (str): Optional argument to use a nested dict in the dict list of the jsonl, 
    instead of the root dict. this can be the case if the above fields are present in each sample, 
    but potentially nested in a field entitled "metadata" for example.

    sub_key (str): Optional argument to use a nested dict in the dict list of the jsonl, 
    instead of the root dict. this can be the case if the above fields are present in each sample, 
    but potentially nested in a field entitled "metadata" for example.

    code_start_index (int): Starting index of the codes to run. To be used to evaluate a subset of the jsonl file ! 
    If not set to None, the framework will run on any samples of the input jsonl 
    having an index between code_start_index and code_end_index

    code_end_index (int): Ending index of the codes to run.

    filter_on_problem (str): Filter on a problem name if necessary. 
    Either set to none, or to a string that is a problem name as specified in the field problem_name of the input jsonl

    multiply_samples_factor (int): Number of times to run the framework on each sample. 
    Allows to run the framework several times on each sample of the input jsonl. 
    Especially if you want to do some majority voting to reduce noise for instance.

    input_handler (str): Input handler to use. 
    Whether code comes with a dataclass (that's the case of input being a string input stream, 
    in which case the field dataclass_code has to be present in the input jsonl), 
    or as a function of a class to find and execute !
    Values can be either "with_dataclass" if the code comes with a dataclass, 
    or the name of the class that contains the function to execute 
    (the framework will look for the method to execute automatically, 
    by matching the arguments to the inputs examples).



    *** LOGS/OUTPUTS ***

    log_outputs (bool): Whether to log outputs.
    Can used to log info in the command line.

    save_results (bool): Whether to save results.
    Save explicitely the outputs of the complexity framework in a folder, so that they can be used after execution of the complexity framework.

    skip_saving_full_results (bool): Whether to save the full results, even if save_results is True.
    The full results include all sets of measures. If you want to save space and you only care about the pure complexity outputs,
    then you can save lots of space by not keeping these full results !

    results_folder_name_root (str): Name of the folder to save results, if results are saved



    *** MEASUREMENTS ***

    shuffle_runs (bool): Whether to shuffle runs.
    This boolean shuffle measurements to be done more randomly, 
    helping to reduce measure noise (each worker taking the measures will shuffle as much as it can the order of the codes to evaluate,
    and on which inputs to evaluate them).

    correct_nlogn (bool): Whether to correct for nlogn complexity.
    This boolean helps correct some python optimization concerning sorting, 
    therefore easing the detection of nlogn complexity (that shows up a lot in the case of sorting algorithms).

    multiplier_op (str): Operation to apply when scaling the size of the inputs.
    Can be addition or multiplication. In either case, the scaling factor is multiplier_mult_step. 
    If multiplier_op is multiplication, then multiplier_mult_step being 2 means the input size is multiplied by 2 at every scaling iteration. 
    If it is addition, it means size will not scale exponentially but rather linearly.

    multiplier_start (int): Initial scaling factor.
    multiplier_start is the initial scaling factor applied on the initial size (as input to the framework) of the input examples. 
    If 1, then the first measure will be done with the initial size of the inputs as given. 
    But higher values enable to directly do measures with larger sizes of inputs.

    multiplier_repeat (int): Number of times to repeat measures.
    Repetition of measures to take on the same size of inputs. 
    1 means no repetition is applied, meaning quick runs of the framework but potentially more noise. 
    Ideally, a deterministic execution would allow using 1. But with a real cpu and real runtimes, 
    higher values help reduce the noise of the measures.

    multiplier_end (int): Highest value of the multiplier.
    3000 means largest inputs will be (initial size of the input) (+ if multiplier_op == addition else x) 3000.

    multiplier_mult_step (int): Step size of the multiplication of input values. 

    multiplier_max_increase (int): multiplier_max_increase enables to limit the step size when increasing the input size. 
    Even if scaling is exponential (when multiplier_op is multiplication), 
    multiplier_max_increase enables to limit the sizes of the steps at some point. 
    So that the start of the measures can be exponential until step size reaches a certain value, 
    after which inputs are further scaled linearly with this value as a fixed step.

    size_of_other_arguments (int): This is the fix size of the other arguments 
    when an argument is being scaled to take measures of the runtime/memory footprint of the code when this input argument is scaled.

    time_profiler (str): Type of time_profiler to use. We recommend using cprofilearound, 
    other profilers can be modified or implemented in the code (we do not garantee their proper execution).



    *** COMPLEXITY FITTING ***

    filter_outliers (bool): Beta version of a filtering method to remove any outlier measures, 
    in the attempt to reduce noise of the measures. This was not used in our evaluation runs, 
    but can be improved to hopefully bring benefits

    apply_penalty (bool): Apply penalties to the different complexity classes during the curve fitting. 
    The penalty coefficient are hard-coded in the code below, 
    as these are supposed to be chosen by careful sweeping and not to be changed across evaluation runs.

    apply_constraints (bool): if apply_constraints, then the curve fitting is done using scipy.optimize.nnls (non negative least squares), 
    otherwise the fitting falls back to using np.linalg.lstsq.

    zero_out_first_value (bool): zero_out_first_value enables to remove constant offset in the measures 

    piecewise_fit (bool): piecewise_fit used to zero out any variations for the small values of inputs that can alter the curve fitting.

    aggregate_y_values (str): used to aggregate measure values accross multiple runs when multiplier_mult_step > 1.

    max_time_rate (float): used for aggregating accross different expansion methods of the inputs.

    elect_complexity_time (str): used for aggregating accross different expansion methods of the inputs, for time complexity.

    elect_complexity_space (str): used for aggregating accross different expansion methods of the inputs, for space complexity.

    fix_constant_complexity (bool): used for aggregating accross different expansion methods of the inputs.

    fix_negligeable_complexity (bool): used for aggregating accross different expansion methods of the inputs.



    *** OUT OF MEM/TIME HANDLING ***

    temp_file_name_seed (str): used by the workers taking the measurements to offload memory to hardware memory, 
    instead of overfilling the RAM.

    memory_limit (int): can cancel a sandbox that uses to much memory space 
    (used to prevent overfilling the memory and therefore making the framework crash).

    timeout (int): various levels of timeouts used in the workers taking the measurements, 
    so to abort executions taking too much time and slowing down the general run.

    large_timeout (int): various levels of timeouts used in the workers taking the measurements, 
    so to abort executions taking too much time and slowing down the general run.

    giga_timeout (int): various levels of timeouts used in the workers taking the measurements, 
    so to abort executions taking too much time and slowing down the general run.

    global_timeout (int): various levels of timeouts used in the workers taking the measurements, 
    so to abort executions taking too much time and slowing down the general run.



    *** CPU ALLOCATIONS ***

    On a single node, all cpus can be used to run measurements on the various codes with various 
    corresponding inputs (different sizes and different expansion methods). In order to reduce noise, 
    workers that take the measures can be contained to certain cpus, 
    or on the contrary be assigned freely to different cpus over the course of their measurements.
    Below are several arguments that can help control this.

    max_workers (int): max_workers is the maximum number of workers that can run concurently on a node. 
    This number will be capted in any case by the total number of cpus on a node, 
    so a very high value corresponds to choosing the default number of cpus available on the node.

    number_solutions_per_worker (int): Each worker can  be used to take measurement on at least one sample code, 
    on all types of inputs (of different sizes) necessary to obtain the complexity value. 
    But we can also choose to run several sample codes (and the corresponding multiple values of inputs) on the same worker, 
    instead of killing it and launching a new one when going to the next sample code.

    main_process_cpu_id_list (str): main_process_cpu_id_list can be none if the main process of the complexity framework 
    can be run freely on any cpus, or be set to a specific cpu ID to contain it on this cpu. 
    It can be also set to a list of cpu ids of the form '0,1,2' if the main process is to be run on cpu IDS 0, 1 and 2.

    forkserver_type (str): forkserver_type can be used to investigate alternative types of forkservers, 
    used to launch sandboxes where workers take measurements. standard is a good forkserver.

    use_distinct_forkservers (bool): use_distinct_forkservers can be used to leverage a different forkserver 
    to launch sandboxes in each process.

    forkserver_cpu_id_list (str): forkserver_cpu_id_list similar to the main process cpu id list, but to host the forkserver.

    sandbox_cpu_id_step (int): step to take among the ordered list of cpu ids (if 10 cpus are available, 
    list of ids is 0,1,2,3,4,5,6,7,8,9) to assign the cpus to the sandboxes. 
    First step is skipped, so to be potentially assigned to the main process and the forkserver. 
    If sandbox_cpu_id_step == 2, then the first worker to take measurements will have access to cpus 2,3, 
    the second worker to cpus 4,5 and so on.

    sandbox_incremental_cpu_id_list (str): among each set of cpus made available to a worker through sandbox_cpu_id_step, 
    we can select a subset using this argument. For instance, using the above example of sandbox_cpu_id_step,
    if sandbox_incremental_cpu_id_list == '0', then the first worker will only work with cpu ID 2. 
    If sandbox_incremental_cpu_id_list is '0,1', then it will work with cpu IDs 2 and 3.

    distinct_forkservers_incremental_cpu_id_list (str): can be used to assign cpus to distinct forkservers 
    if use_distinct_forkservers is set to True, among the cpus made available for the sandbox of each worker. 
    In the above example, if sandbox_incremental_cpu_id_list = "0" and distinct_forkservers_incremental_cpu_id_list = "1", 
    then the first worker will use cpu ID 2 to launch the sandbox and cpu ID 3 to handle the forkserver that launches the sandbox.
"""

    current_datetime = datetime.datetime.now()

    # Hard coded parameters of the different classes of the curve fitting !
    complexity_name_function_list_time = [
        ('o(1)', Constant, 1.0, 0),
        ('o(1)', Constant_2, 1.0, 0),
        ('o(logn)', Logarithmic, 5.0, 0.5),
        ('o(n)', Linear, 4.5, 1),
        ('o(nlogn)', Linearithmic, 4.5, 1.5),
        ('o(n^2)', Quadratic, 3.5, 2),
        ('o(n^3)', Cubic, 1000000, 3),
    ]

    complexity_name_function_list_space = [
        ('o(1)', Constant, 1.0, 0),
        ('o(1)', Constant_2, 1.0, 0),
        ('o(logn)', Logarithmic, 5.0, 0.5),
        ('o(n)', Linear, 1.5, 1),
        ('o(nlogn)', Linearithmic, 5.0, 1.5),
        ('o(n^2)', Quadratic, 2.0, 2),
        ('o(n^3)', Cubic, 1000000, 3),
    ]

    # Processing input arguments

    current_process = psutil.Process(os.getpid())
    cpu_id_list = current_process.cpu_affinity()
    number_physical_cpus = int(psutil.cpu_count(logical=False))

    temp_file_name_seed += str(random.randint(1, 10000000000000))

    if main_process_cpu_id_list is None:
        # We do not restict the cpus on which the main thread runs
        print('Running main process on all cpus')
        pass

    else:
        main_process_cpu_id_list = [
            int(x) for x in main_process_cpu_id_list.split(',')]
        print('Running main process on following cpus', main_process_cpu_id_list)
        current_process.cpu_affinity(main_process_cpu_id_list)

    if forkserver_cpu_id_list is None:
        # We do not restict the cpus on which the main thread runs
        print('Running forkserver process on all cpus')
        pass

    else:
        forkserver_cpu_id_list = [int(x)
                                  for x in forkserver_cpu_id_list.split(',')]
        print('Running forkserver process on following cpus',
              forkserver_cpu_id_list)

    assert forkserver_type in ["standard", "custom"]

    if forkserver_type == "standard":
        from sandbox.server_runtimes import Executor, ForkServer, ResourceLimits, JSONConnection, Executor
        bubblewrap_executor = Executor.BUBBLEWRAP

    elif forkserver_type == "custom":
        from sandbox.server_runtimes_custom import Executor, ForkServer, ResourceLimits, JSONConnection, Executor
        bubblewrap_executor = Executor.BUBBLEWRAPUNLOCK

    else:
        raise Exception('not supported !')

    if not use_distinct_forkservers:
        fs = ForkServer(forkserver_cpu_id_list)
        print('fork server ready')
    else:
        fs = None

    sandbox_cpu_id_list = [
        cpu_id for cpu_id in cpu_id_list
        if (
            (
                main_process_cpu_id_list is None
                or
                (cpu_id not in main_process_cpu_id_list)
            )
            and
            (
                forkserver_cpu_id_list is None
                or
                (cpu_id not in forkserver_cpu_id_list)
            )
            and
            (
                cpu_id % sandbox_cpu_id_step == 0
            )
        )
    ]

    if distinct_forkservers_incremental_cpu_id_list is None:
        # We do not restict the cpus on which the main thread runs
        print('Running distinct forkserver (if any) processes on all cpus')
        def distinct_forkservers_cpu_id_list_function(x): return None

    else:
        distinct_forkservers_incremental_cpu_id_list = [
            int(x) for x in distinct_forkservers_incremental_cpu_id_list.split(',')]
        print('Running distinct forkserver (if any) on following cpus',
              distinct_forkservers_incremental_cpu_id_list)

        def distinct_forkservers_cpu_id_list_function(
            x): return [x + y for y in distinct_forkservers_incremental_cpu_id_list if (x+y) in cpu_id_list]

    if sandbox_incremental_cpu_id_list is None:
        # We do not restict the cpus on which the main thread runs
        print('Running sandbox processes on all cpus')
        def sandbox_cpu_id_list_function(x): return None

    else:
        sandbox_incremental_cpu_id_list = [
            int(x) for x in sandbox_incremental_cpu_id_list.split(',')]
        print('Running sandbox processes on following cpus',
              sandbox_incremental_cpu_id_list)

        def sandbox_cpu_id_list_function(
            x): return [x + y for y in sandbox_incremental_cpu_id_list if (x+y) in cpu_id_list]

    max_workers = min(len(sandbox_cpu_id_list), max_workers)
    print('Number of workers', max_workers)
    print('sandbox_cpu_id_list', sandbox_cpu_id_list)

    assert len(sandbox_cpu_id_list) >= max_workers

    print('code_end_index', code_end_index)

    assert code_start_index is not None
    assert (code_end_index is None) or (code_start_index < code_end_index)

    max_num_batches_by_group = number_solutions_per_worker * max_workers

    count_memory_errors = 0

    solution_dataset = []
    full_solution_dataset = []

    if path_to_jsonl_file is None:
        print('No file to read, path_to_jsonl_file is None, exiting...')
        return

    with open(Path(path_to_jsonl_file), 'r') as jsonl_input:
        for i, entry in enumerate(jsonl_input):

            if code_end_index is None:
                if i < code_start_index:
                    continue
            else:
                if i not in range(code_start_index, code_end_index, 1):
                    continue

            solution = (
                json.loads(entry) if sub_key is None
                else json.loads(entry)[sub_key]
            )

            full_solution = json.loads(entry)

            if filter_on_problem is not None:
                if solution['question_name'] != filter_on_problem:
                    continue

            for attribute_name_in_code, attribute_name_outside in zip(
                [
                    "variable_name_to_input_dict",
                    "question_name",
                    "code_content",
                    "time",
                    "space",
                    "dataclass_code",
                ],
                [
                    "inputs_example",
                    "problem_name",
                    "solution_code",
                    "time_complexity_inferred",
                    "space_complexity_inferred",
                    "dataclass_code",
                ]
            ):
                if attribute_name_outside not in solution.keys():
                    solution[attribute_name_in_code] = ""

                else:
                    solution[attribute_name_in_code] = solution[attribute_name_outside]

            for _ in range(multiply_samples_factor):
                solution_dataset.append(solution)
                full_solution_dataset.append(full_solution)

    num_batches = len(solution_dataset)
    assert len(full_solution_dataset) == len(solution_dataset)

    all_question_inputs_list = []
    all_full_question_list = []
    all_runtime_dict_list = []
    all_memory_dict_list = []
    all_error_list = []

    total_execution_time = time.time()

    print('in total we have ', len(solution_dataset), 'solutions to examine')

    print('################### STEP 1: taking measures of runtime and memory footprint !')

    for group_batch_index in range(0, num_batches//max_num_batches_by_group + 1):
        print("###################################")
        print('group_batch_index', group_batch_index)

        if len(
            solution_dataset[group_batch_index * max_num_batches_by_group:min(
                (group_batch_index + 1) * max_num_batches_by_group, num_batches)]
        ) == 0:
            continue

        question_inputs_list = []
        full_question_list = []
        runtime_dict_list = []
        memory_dict_list = []
        error_list = []
        index_list = []

        # assert number_physical_cpus >= max_workers

        for i, (solution, full_solution) in enumerate(zip(
            solution_dataset[group_batch_index * max_num_batches_by_group:min(
                (group_batch_index + 1) * max_num_batches_by_group, num_batches)],
            full_solution_dataset[group_batch_index * max_num_batches_by_group:min(
                (group_batch_index + 1) * max_num_batches_by_group, num_batches)],
        )):
            question_inputs_list.append(
                {
                    'variable_name_to_input_dict': solution['variable_name_to_input_dict'],
                    'question_name': solution['question_name'],
                    'code_content': solution['code_content'],
                    'time_complexity': map_true_complexity(solution['time']) if type(solution['time']) == str else '',
                    'space_complexity': map_true_complexity(solution['space']) if type(solution['space']) == str else '',
                    'dataclass_code': solution['dataclass_code'] if input_handler == 'with_dataclass' else None,
                }
            )
            full_question_list.append(
                full_solution
            )

        count_solutions = 0
        batch_size = math.ceil(len(question_inputs_list) / max_workers)

        fs_list = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            future_to_index_dict = {}

            for i in range(0, max_workers, 1):
                if len(question_inputs_list[i * batch_size: (i+1) * batch_size]) == 0:
                    continue

                # print('launching process', i)

                if use_distinct_forkservers:
                    temp_fs = ForkServer(
                        cpu_id_list=distinct_forkservers_cpu_id_list_function(
                            sandbox_cpu_id_list[i]),
                    )
                    fs_list.append(temp_fs)

                future = (
                    executor.submit(
                        exec_python_time_space_complexities,
                        question_inputs_list=question_inputs_list[i * batch_size: (
                            i+1) * batch_size],
                        multiplier_start=multiplier_start,
                        multiplier_repeat=multiplier_repeat,
                        multiplier_end=multiplier_end,
                        multiplier_mult_step=multiplier_mult_step,
                        multiplier_max_increase=multiplier_max_increase,
                        time_profiler=time_profiler,
                        print_debugging_dict_list=False,
                        correct_nlogn=correct_nlogn,
                        multiplier_op=multiplier_op,
                        shuffle_runs=shuffle_runs,
                        timeout=timeout,
                        large_timeout=large_timeout,
                        giga_timeout=giga_timeout,
                        global_timeout=global_timeout,
                        memory_limit=memory_limit,
                        size_of_other_arguments=size_of_other_arguments,
                        temp_file_name_seed=temp_file_name_seed +
                        '_{}_{}'.format(group_batch_index, i),
                        fork_server=temp_fs if use_distinct_forkservers else fs,
                        bubblewrap_executor=bubblewrap_executor,
                        cpu_id_list=sandbox_cpu_id_list_function(
                            sandbox_cpu_id_list[i]),
                        forkserver_type=forkserver_type,
                        input_handler=input_handler,
                    )
                )

                future_to_index_dict[future] = i
                futures.append(future)

            # print('finished launching all processess')
            # print('total of', len(futures), 'processes to wait for')

            for f in as_completed(futures):  # , timeout=1

                print('processing process id {}'.format(
                    future_to_index_dict[f]))

                response = f.result()
                runtime_dict_list_temp, memory_dict_list_temp, error_list_temp, _ = response

                runtime_dict_list.append(runtime_dict_list_temp)
                count_solutions += len(runtime_dict_list_temp)
                memory_dict_list.append(memory_dict_list_temp)

                if type(error_list_temp) == list:
                    error_list.append(error_list_temp)
                else:
                    error_list.append([error_list_temp])

                temp_error_list = error_list[-1]
                if type(temp_error_list) == str:
                    temp_error_list = [temp_error_list]
                elif type(temp_error_list) != list:
                    temp_error_list = [str(temp_error_list)]

                # print(temp_error_list)

                if any(['MemoryError' in str(temp_error) for temp_error in temp_error_list]):
                    print(
                        '################ MEMORY ERROR: you might want to change memory_limit')
                    print(
                        future_to_index_dict[f]
                    )
                    print(question_inputs_list[future_to_index_dict[f] *
                          batch_size: (future_to_index_dict[f]+1) * batch_size])
                    print(temp_error_list)

                index_list.append(future_to_index_dict[f])

                # print('Finished process id {}'.format(future_to_index_dict[f]), 'remaining processes are', len(futures) - len(index_list))

        print('Finished all processess')
        print('Received', len(index_list), 'responses')

        for temp_fs in fs_list:
            temp_fs.stop()

        argsort_list = np.argsort(index_list)

        error_list_reordered = []
        runtime_dict_list_reordered = []
        memory_dict_list_reordered = []
        question_inputs_list_reordered = []
        full_question_list_reordered = []

        for argsort_ in argsort_list:
            try:
                assert len(question_inputs_list[index_list[argsort_] * batch_size: (
                    index_list[argsort_]+1) * batch_size]) == len(error_list[argsort_])
                assert len(full_question_list[index_list[argsort_] * batch_size: (
                    index_list[argsort_]+1) * batch_size]) == len(error_list[argsort_])
                assert len(error_list[argsort_]) == len(
                    runtime_dict_list[argsort_])
                assert len(error_list[argsort_]) == len(
                    memory_dict_list[argsort_])

                assert len(error_list[argsort_]) > 0

                assert type(error_list[argsort_]) == list
                assert type(runtime_dict_list[argsort_]) == list
                assert type(memory_dict_list[argsort_]) == list
                assert type(question_inputs_list[index_list[argsort_] * batch_size: (
                    index_list[argsort_]+1) * batch_size]) == list
                assert type(full_question_list[index_list[argsort_] * batch_size: (
                    index_list[argsort_]+1) * batch_size]) == list

            except Exception as e:
                # print('error')
                # print(error_list[argsort_])
                temp_error_list = error_list[argsort_]
                if type(temp_error_list) == str:
                    temp_error_list = [temp_error_list]
                elif type(temp_error_list) != list:
                    temp_error_list = [str(temp_error_list)]

                if any(['MemoryError' in str(temp_error) for temp_error in temp_error_list]):
                    count_memory_errors += 1

                length = len(
                    question_inputs_list[index_list[argsort_] * batch_size: (index_list[argsort_]+1) * batch_size])

                error_list[argsort_] = [f'error: {str(e) + str(error_list)}'] * length
                runtime_dict_list[argsort_] = [{}] * length
                memory_dict_list[argsort_] = [{}] * length

            for error in error_list[argsort_]:
                error_list_reordered.append(error)

            for runtime_dict in runtime_dict_list[argsort_]:
                runtime_dict_list_reordered.append(runtime_dict)

            for memory_dict in memory_dict_list[argsort_]:
                memory_dict_list_reordered.append(memory_dict)

            for question_inputs in question_inputs_list[index_list[argsort_] * batch_size: (index_list[argsort_]+1) * batch_size]:
                question_inputs_list_reordered.append(question_inputs)

            for full_question in full_question_list[index_list[argsort_] * batch_size: (index_list[argsort_]+1) * batch_size]:
                full_question_list_reordered.append(full_question)

        error_list = error_list_reordered
        runtime_dict_list = runtime_dict_list_reordered
        memory_dict_list = memory_dict_list_reordered
        question_inputs_list = question_inputs_list_reordered
        full_question_list = full_question_list_reordered

        assert len(runtime_dict_list) == len(question_inputs_list)
        assert len(memory_dict_list) == len(question_inputs_list)
        assert len(error_list) == len(question_inputs_list)
        assert len(full_question_list) == len(question_inputs_list)

        for x in question_inputs_list:
            all_question_inputs_list.append(x)

        for x in full_question_list:
            all_full_question_list.append(x)

        for x in runtime_dict_list:
            all_runtime_dict_list.append(x)

        for x in memory_dict_list:
            all_memory_dict_list.append(x)

        for x in error_list:
            all_error_list.append(x)

        print('going to next batch')

    if not use_distinct_forkservers:
        fs.stop()

    print('Stopped fork')
    print('to process we have', len(all_question_inputs_list))

    non_timeout_ratio = len(all_question_inputs_list) / len(solution_dataset)

    print('################### STEP 2: complexity curve fitting !')

    import collections

    parsed_time_complexity_list = []
    time_max_coeff_list = []
    time_complexity_found_peak_list = []
    count_has_infered_time_complexity = 0

    parsed_space_complexity_list = []
    space_max_coeff_list = []
    space_complexity_found_peak_list = []
    count_has_infered_space_complexity = 0

    count_time_has_peak = 0
    count_space_has_peak = 0

    # In runtime_dict_list, how do each element look like ?
    # {
    #     (variable_name_1, variable_type_1, dimension_1): {
    #         multiplier_value_1: [
    #             {
    #                 'runtime_list': [value_1, value_2, value_3, value_4, value_5],
    #                 'id_': 'a_descriptive_id_for_the_run',
    #                 'tag_list': [(variable_name_1, variable_type_1, dimension_1)],
    #                 'priority': priority, # int, the lower the higher the priority when computing the runtime
    #             },
    #             {
    #                 'runtime_list': [value_1, value_2, value_3, value_4, value_5],
    #                 'id_': id_,
    #                 'tag_list': [(variable_name_1, variable_type_1, dimension_1),(variable_name_2, variable_type_2, dimension_2)],
    #                 'priority': priority,
    #             },
    #             ...
    #         ],
    #         multiplier_value_2: [
    #             ...
    #         ],
    #         multiplier_value_3: [
    #             ...
    #         ],
    #         multiplier_value_4: [
    #             ...
    #         ],
    #         multiplier_value_5: [
    #             ...
    #         ],
    #     },
    #     (variable_name_2, variable_type_2, dimension_2): {
    #         ...
    #     },
    #     (variable_name_3, variable_type_3, dimension_3): {
    #         ...
    #     }
    # }
    #
    # OR IN A DIFFERENT MANNER, below how this dictionnary is constructed in the sandbox
    #
    # runtime_dict_list[index][(variable_name, variable_type, dimension)][multiplier_current].append(
    #     {
    #         'runtime_list': temp_runtime_list,
    #         'id_': id_,
    #         'tag_list': tag_list,
    #         'priority': priority,
    #     }
    # )

    runtime_dict_anonymized_list = []
    memory_dict_anonymized_list = []

    runtime_variable_name_type_dimension_to_index_list = []
    memory_variable_name_type_dimension_to_index_list = []

    for question_inputs, runtime_dict, memory_dict, error in zip(
        all_question_inputs_list, all_runtime_dict_list, all_memory_dict_list, all_error_list
    ):
        variable_name_type_dimension_to_index = {}
        variable_name_type_dimension_to_index_anonymized = {}
        variable_name_to_index_dict = {}
        variable_name_to_index_dict_count = 0

        for i, variable_name_type_dimension in enumerate(runtime_dict.keys()):
            variable_name_type_dimension_to_index[variable_name_type_dimension] = i

            if variable_name_type_dimension.split('####')[0] not in variable_name_to_index_dict:
                variable_name_to_index_dict[variable_name_type_dimension.split(
                    '####')[0]] = str(variable_name_to_index_dict_count)
                variable_name_to_index_dict_count += 1

            variable_name_type_dimension_to_index_anonymized[
                '####'.join(
                    [variable_name_to_index_dict[variable_name_type_dimension.split('####')[
                        0]]]
                    +
                    variable_name_type_dimension.split('####')[-2:]
                )
            ] = i

        runtime_dict_anonymized = dict()
        runtime_variable_name_type_dimension_to_index_list.append(
            variable_name_type_dimension_to_index_anonymized)

        try:
            for variable_name_type_dimension in runtime_dict.keys():
                runtime_dict_anonymized[variable_name_type_dimension_to_index[variable_name_type_dimension]] = dict(
                )
                for multiplier_value in runtime_dict[variable_name_type_dimension].keys():
                    runtime_dict_anonymized[variable_name_type_dimension_to_index[variable_name_type_dimension]][multiplier_value] = list(
                    )
                    for runtime_details in runtime_dict[variable_name_type_dimension][multiplier_value]:
                        runtime_dict_anonymized[variable_name_type_dimension_to_index[variable_name_type_dimension]][multiplier_value].append(
                            {
                                'value_list': runtime_details['value_list'],
                                'id_': runtime_details['id_'],
                                'tag_list': list(map(lambda x: variable_name_type_dimension_to_index[x], runtime_details['tag_list'])),
                                'priority': runtime_details['priority'],
                            }
                        )
        except Exception as e:
            # print('little error')
            pass

        try:
            (
                parsed_time_complexity,
                time_complexity_found_peak,
                time_complexity_coeff_product,
                time_complexity_coeff_max,
                found_peak,
            ) = infer_complexity_from_values(
                value_dict=runtime_dict,
                complexity_name_function_list=complexity_name_function_list_time,
                filter_outliers=filter_outliers,
                apply_penalty=apply_penalty,
                apply_constraints=apply_constraints,
                zero_out_first_value=zero_out_first_value,
                piecewise_fit=piecewise_fit,
                aggregate_y_values=aggregate_y_values,
                max_time_rate=max_time_rate,
                elect_complexity=elect_complexity_time,
                fix_constant_complexity=fix_constant_complexity,
                fix_negligeable_complexity=fix_negligeable_complexity,
                enlarge_values=False,
                print_info=False,
                multiplier_start=multiplier_start,
                aggressive_max_time_x_scaling=True,
            )
        except Exception as e:
            # print(runtime_dict_anonymized)
            (
                parsed_time_complexity,
                time_complexity_found_peak,
                time_complexity_coeff_product,
                time_complexity_coeff_max,
                found_peak,
            ) = (None, False, None, None, None)

        if found_peak is not None:
            count_time_has_peak += int(found_peak)

        if parsed_time_complexity is None:
            time_complexity_is_success = False

        else:
            time_complexity_is_success = (
                equality_complexities(
                    parsed_time_complexity, question_inputs['time_complexity'])
            )

        # Space complexity

        variable_name_type_dimension_to_index = {}
        variable_name_type_dimension_to_index_anonymized = {}
        variable_name_to_index_dict = {}
        variable_name_to_index_dict_count = 0

        for i, variable_name_type_dimension in enumerate(memory_dict.keys()):
            variable_name_type_dimension_to_index[variable_name_type_dimension] = i

            if variable_name_type_dimension.split('####')[0] not in variable_name_to_index_dict:
                variable_name_to_index_dict[variable_name_type_dimension.split(
                    '####')[0]] = str(variable_name_to_index_dict_count)
                variable_name_to_index_dict_count += 1

            variable_name_type_dimension_to_index_anonymized[
                '####'.join(
                    [variable_name_to_index_dict[variable_name_type_dimension.split('####')[
                        0]]]
                    +
                    variable_name_type_dimension.split('####')[-2:]
                )
            ] = i

        memory_dict_anonymized = dict()
        memory_variable_name_type_dimension_to_index_list.append(
            variable_name_type_dimension_to_index_anonymized)

        try:
            for variable_name_type_dimension in memory_dict.keys():
                memory_dict_anonymized[variable_name_type_dimension_to_index[variable_name_type_dimension]] = dict(
                )
                for multiplier_value in memory_dict[variable_name_type_dimension].keys():
                    memory_dict_anonymized[variable_name_type_dimension_to_index[variable_name_type_dimension]][multiplier_value] = list(
                    )
                    for memory_details in memory_dict[variable_name_type_dimension][multiplier_value]:
                        memory_dict_anonymized[variable_name_type_dimension_to_index[variable_name_type_dimension]][multiplier_value].append(
                            {
                                'value_list': memory_details['value_list'],
                                'id_': memory_details['id_'],
                                'tag_list': list(map(lambda x: variable_name_type_dimension_to_index[x], memory_details['tag_list'])),
                                'priority': memory_details['priority'],
                            }
                        )
        except Exception as e:
            # print('space little error')
            pass

        try:
            (
                parsed_space_complexity,
                space_complexity_found_peak,
                space_complexity_coeff_product,
                space_complexity_coeff_max,
                found_peak_bis,
            ) = infer_complexity_from_values(
                value_dict=memory_dict,
                complexity_name_function_list=complexity_name_function_list_space,
                filter_outliers=filter_outliers,
                apply_penalty=apply_penalty,
                apply_constraints=apply_constraints,
                zero_out_first_value=zero_out_first_value,
                piecewise_fit=piecewise_fit,
                aggregate_y_values=aggregate_y_values,
                max_time_rate=max_time_rate,
                elect_complexity=elect_complexity_space,
                fix_constant_complexity=False,
                fix_negligeable_complexity=fix_negligeable_complexity,
                enlarge_values=False,
                print_info=False,
                multiplier_start=multiplier_start,
                aggressive_max_time_x_scaling=True,
            )
        except Exception as e:
            # print(runtime_dict_anonymized)
            # print(memory_dict_anonymized)
            (
                parsed_space_complexity,
                space_complexity_found_peak,
                space_complexity_coeff_product,
                space_complexity_coeff_max,
                found_peak_bis,
            ) = (None, False, None, None, None)

        if found_peak_bis is not None:
            count_space_has_peak += int(found_peak_bis)

        if parsed_space_complexity is None:
            space_complexity_is_success = False

        else:
            space_complexity_is_success = (
                equality_complexities(
                    parsed_space_complexity, question_inputs['space_complexity'])
            )

        if log_outputs:
            print('################')
            print(question_inputs['question_name'])
            print(runtime_dict.keys())
            print("####")
            print(runtime_dict_anonymized)
            print("####")
            print(memory_dict_anonymized)
            print(error)
            print(question_inputs['code_content'].replace('\\n', '\n'))
            print('found_peak', found_peak)
            print(
                time_complexity_is_success,
                question_inputs['time_complexity'],
                parsed_time_complexity,
                time_complexity_coeff_product,
                time_complexity_coeff_max,
            )
            print(
                space_complexity_is_success,
                question_inputs['space_complexity'],
                parsed_space_complexity,
                space_complexity_coeff_product,
                space_complexity_coeff_max,
            )

        if parsed_time_complexity is not None:
            count_has_infered_time_complexity += 1

        parsed_time_complexity_list.append(parsed_time_complexity)
        time_max_coeff_list.append(
            float(time_complexity_coeff_max)
            if time_complexity_coeff_max is not None
            else None
        )
        time_complexity_found_peak_list.append(time_complexity_found_peak)
        runtime_dict_anonymized_list.append(runtime_dict_anonymized)

        if parsed_space_complexity is not None:
            count_has_infered_space_complexity += 1

        parsed_space_complexity_list.append(parsed_space_complexity)
        space_max_coeff_list.append(
            float(space_complexity_coeff_max)
            if space_complexity_coeff_max is not None
            else space_complexity_coeff_max
        )
        space_complexity_found_peak_list.append(space_complexity_found_peak)

        memory_dict_anonymized_list.append(memory_dict_anonymized)

    has_infered_time_complexity_ratio = count_has_infered_time_complexity / \
        len(solution_dataset)
    has_infered_space_complexity_ratio = count_has_infered_space_complexity / \
        len(solution_dataset)

    time_has_peak_ratio = count_time_has_peak / len(solution_dataset)
    space_has_peak_ratio = count_space_has_peak / len(solution_dataset)

    # SAVING DATA

    arguments_of_framework = {}

    for x, x_name in zip([
        path_to_jsonl_file,
        sub_key,
        code_start_index,
        code_end_index,
        filter_on_problem,
        multiply_samples_factor,
        input_handler,
        log_outputs,
        save_results,
        skip_saving_full_results,
        results_folder_name_root,
        shuffle_runs,
        correct_nlogn,
        multiplier_op,
        multiplier_start,
        multiplier_repeat,
        multiplier_end,
        multiplier_mult_step,
        multiplier_max_increase,
        size_of_other_arguments,
        time_profiler,
        filter_outliers,
        apply_penalty,
        apply_constraints,
        zero_out_first_value,
        piecewise_fit,
        aggregate_y_values,
        max_time_rate,
        elect_complexity_time,
        elect_complexity_space,
        fix_constant_complexity,
        fix_negligeable_complexity,
        temp_file_name_seed,
        memory_limit,
        timeout,
        large_timeout,
        giga_timeout,
        global_timeout,
        max_workers,
        number_solutions_per_worker,
        main_process_cpu_id_list,
        forkserver_type,
        use_distinct_forkservers,
        forkserver_cpu_id_list,
        sandbox_cpu_id_step,
        sandbox_incremental_cpu_id_list,
        distinct_forkservers_incremental_cpu_id_list,
        slurm_array_task_id,
        slurm_array_task_max,
        slurm_array_task_min,
        slurm_array_job_id,
    ], [
        "path_to_jsonl_file",
        "sub_key",
        "code_start_index",
        "code_end_index",
        "filter_on_problem",
        "multiply_samples_factor",
        "input_handler",
        "log_outputs",
        "save_results",
        "skip_saving_full_results",
        "results_folder_name_root",
        "shuffle_runs",
        "correct_nlogn",
        "multiplier_op",
        "multiplier_start",
        "multiplier_repeat",
        "multiplier_end",
        "multiplier_mult_step",
        "multiplier_max_increase",
        "size_of_other_arguments",
        "time_profiler",
        "filter_outliers",
        "apply_penalty",
        "apply_constraints",
        "zero_out_first_value",
        "piecewise_fit",
        "aggregate_y_values",
        "max_time_rate",
        "elect_complexity_time",
        "elect_complexity_space",
        "fix_constant_complexity",
        "fix_negligeable_complexity",
        "temp_file_name_seed",
        "memory_limit",
        "timeout",
        "large_timeout",
        "giga_timeout",
        "global_timeout",
        "max_workers",
        "number_solutions_per_worker",
        "main_process_cpu_id_list",
        "forkserver_type",
        "use_distinct_forkservers",
        "forkserver_cpu_id_list",
        "sandbox_cpu_id_step",
        "sandbox_incremental_cpu_id_list",
        "distinct_forkservers_incremental_cpu_id_list",
        "slurm_array_task_id",
        "slurm_array_task_max",
        "slurm_array_task_min",
        "slurm_array_job_id",
    ]):
        arguments_of_framework[
            x_name
        ] = str(x)

    if log_outputs:
        print("non_timeout_ratio", non_timeout_ratio)
        print("has_infered_time_complexity_ratio",
              has_infered_time_complexity_ratio)
        print("has_infered_space_complexity_ratio",
              has_infered_space_complexity_ratio)
        print('count_memory_errors', count_memory_errors)
        print('time_has_peak_ratio', time_has_peak_ratio)
        print('space_has_peak_ratio', space_has_peak_ratio)

    if save_results:
        # There are two outputs, one that is light and one that is full

        # Handle the case where there is a master folder because this is slurm array
        os.makedirs(results_folder_name_root, exist_ok=True)

        if (
            slurm_array_task_max is not None
        ) and (
            slurm_array_task_min is not None
        ) and (
            slurm_array_task_id is not None
        ) and (
            slurm_array_job_id is not None
        ):
            existing_folder_list = [
                f for f in os.listdir(results_folder_name_root) if os.path.isdir(os.path.join(results_folder_name_root, f))
            ]

            slurm_array_folder_name = f"results_jobid_{slurm_array_job_id}"

            for x in existing_folder_list:
                if x.startswith(slurm_array_folder_name):
                    slurm_array_folder_name = x
                    break

            else:
                # creating the root folder 
                slurm_array_folder_name += f"_datetime_{current_datetime.strftime('%Y%m%d_%H%M%S')}_id_{random.randint(10 ** 10, 10 ** 20)}" 

                if not os.path.isdir(os.path.join(results_folder_name_root, slurm_array_folder_name)):
                    os.makedirs(os.path.join(results_folder_name_root, slurm_array_folder_name), exist_ok=False)
                else:
                    raise Exception('Failed to save')

            results_folder_name_root = os.path.join(
                results_folder_name_root,
                slurm_array_folder_name,
            )

            folder_name = f"results_taskid_{slurm_array_task_id}_datetime_{current_datetime.strftime('%Y%m%d_%H%M%S')}_id_{random.randint(10 ** 10, 10 ** 20)}"

        else:

            folder_name = f"results_datetime_{current_datetime.strftime('%Y%m%d_%H%M%S')}_id_{random.randint(10 ** 10, 10 ** 20)}"

        if not os.path.isdir(os.path.join(results_folder_name_root, folder_name)):
            os.makedirs(os.path.join(results_folder_name_root, folder_name), exist_ok=False)

        else:
            raise Exception('failed to save')

        complexity_labels_light = []
        complexity_labels_full = []
        additional_logs = []

        for (
            all_full_question,
            time_complexity,
            space_complexity,
            time_coeff,
            space_coeff,
            runtime_measures,
            memory_footprint_measures,
            runtime_variable_name_type_dimension_to_index,
            memory_variable_name_type_dimension_to_index,
            error_content,
            time_peak,
            space_peak,
        ) in zip(
            all_full_question_list,
            parsed_time_complexity_list,
            parsed_space_complexity_list,
            time_max_coeff_list,
            space_max_coeff_list,
            runtime_dict_anonymized_list,
            memory_dict_anonymized_list,
            runtime_variable_name_type_dimension_to_index_list,
            memory_variable_name_type_dimension_to_index_list,
            all_error_list,
            time_complexity_found_peak_list,
            space_complexity_found_peak_list,
        ):
            complexity_labels_light.append(
                {
                    "problem_id": all_full_question["problem_id"],
                    "problem_name": all_full_question["problem_name"],
                    "solution_id": all_full_question["solution_id"],
                    "time_complexity_inferred": correct_complexity_formatting(time_complexity),
                    "space_complexity_inferred": correct_complexity_formatting(space_complexity),
                    "time_curve_coefficient": time_coeff,
                    "space_curve_coefficient": space_coeff,
                    "additional_data_from_input": {
                        key_: value_
                        for key_, value_ in all_full_question.items()
                        if key_ not in ["problem_id", "problem_name", "solution_id"]
                    },
                }
            )

            if not skip_saving_full_results:
                complexity_labels_full.append(
                    {
                        "problem_id": all_full_question["problem_id"],
                        "problem_name": all_full_question["problem_name"],
                        "solution_id": all_full_question["solution_id"],
                        "time_complexity_inferred": correct_complexity_formatting(time_complexity),
                        "space_complexity_inferred": correct_complexity_formatting(space_complexity),
                        "time_curve_coefficient": time_coeff,
                        "space_curve_coefficient": space_coeff,
                        "query_dataclass_code": all_full_question.get("dataclass_code", None),
                        "query_code": all_full_question["solution_code"],
                        "query_inputs_example": all_full_question["inputs_example"],
                        "runtime_measures": convert_measures_dict_format(runtime_measures),
                        "memory_footprint_measures": convert_measures_dict_format(memory_footprint_measures),
                        "runtime_measures_set_id_to_input_properties": convert_measures_set_id_to_input_properties_format(
                            runtime_variable_name_type_dimension_to_index
                        ),
                        "memory_footprint_measures_set_id_to_input_properties": convert_measures_set_id_to_input_properties_format(
                            memory_variable_name_type_dimension_to_index
                        ),
                        "additional_data_from_input": {
                            key_: value_
                            for key_, value_ in all_full_question.items()
                            if key_ not in ["problem_id", "problem_name", "solution_id"]
                        },
                    }
                )

            additional_logs.append(
                {
                    "problem_id": all_full_question["problem_id"],
                    "problem_name": all_full_question["problem_name"],
                    "solution_id": all_full_question["solution_id"],
                    "error_log": error_content,
                    "time_curve_has_a_peak": time_peak,
                    "space_curve_has_a_peak": space_peak,
                }
            )

        with open(os.path.join(
            os.path.join(results_folder_name_root, folder_name), 
            'arguments_of_complexity_framework.json'
        ), 'w') as f:
            json.dump(arguments_of_framework, f)

        with open(os.path.join(
            os.path.join(results_folder_name_root, folder_name), 
            'complexity_labels_light.json'
        ), 'w') as f:
            json.dump(complexity_labels_light, f)

        if not skip_saving_full_results:
            with open(os.path.join(
                os.path.join(results_folder_name_root, folder_name), 
                'complexity_labels_full.json'
            ), 'w') as f:
                json.dump(complexity_labels_full, f)

        with open(os.path.join(
            os.path.join(results_folder_name_root, folder_name), 
            'additional_logs.json'
        ), 'w') as f:
            json.dump(additional_logs, f)

        shutil.make_archive(
            os.path.join(results_folder_name_root, folder_name), 
            'zip',
            results_folder_name_root, 
            folder_name,
        )

        shutil.rmtree(os.path.join(results_folder_name_root, folder_name))
