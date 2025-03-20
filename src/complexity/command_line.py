# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from main import run_complexity_framework
import argparse
from pathlib import Path
import json
import numpy as np
import os


def none_or_str(value):
    if value == 'None' or value == 'none':
        return None

    return str(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_jsonl_file", type=str, required=True)
    parser.add_argument("--sub_key", type=none_or_str,
                        required=False, default=None)
    parser.add_argument("--code_start_index", type=int,
                        required=False, default=0)
    parser.add_argument("--code_end_index", type=int,
                        required=False, default=None)
    parser.add_argument("--filter_on_problem", type=none_or_str,
                        required=False, default=None)
    parser.add_argument("--multiply_samples_factor", type=int,
                        required=False, default=1)
    parser.add_argument("--input_handler", type=str,
                        required=False, default="with_dataclass")
    parser.add_argument('--log_outputs',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_results',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--skip_saving_full_results',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slurm_array_task_id", type=int,
                        required=False, default=None)
    parser.add_argument("--slurm_array_task_max", type=int,
                        required=False, default=None)
    parser.add_argument("--slurm_array_task_min", type=int,
                        required=False, default=None)
    parser.add_argument("--slurm_array_job_id", type=int,
                        required=False, default=None)
    parser.add_argument('--results_folder_name_root', type=str,
                        required=False, default="results")
    parser.add_argument('--shuffle_runs',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--correct_nlogn',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multiplier_op", type=str,
                        required=False, default="multiplication")
    parser.add_argument("--multiplier_start", type=int,
                        required=False, default=1)
    parser.add_argument("--multiplier_repeat", type=int,
                        required=False, default=10)
    parser.add_argument("--multiplier_end", type=int,
                        required=False, default=3100)
    parser.add_argument("--multiplier_mult_step", type=int,
                        required=False, default=2)
    parser.add_argument("--multiplier_max_increase",
                        type=int, required=False, default=512)
    parser.add_argument("--size_of_other_arguments",
                        type=int, required=False, default=1000)
    parser.add_argument("--time_profiler", type=str,
                        required=False, default="cprofilearound")
    parser.add_argument('--filter_outliers',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--apply_penalty',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--apply_constraints',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--zero_out_first_value',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--piecewise_fit',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--aggregate_y_values", type=str,
                        required=False, default="most_stable_aggregate_aggregate")
    parser.add_argument("--max_time_rate", type=float,
                        required=False, default=0.8)
    parser.add_argument("--elect_complexity_time", type=str,
                        required=False, default="min")
    parser.add_argument("--elect_complexity_space",
                        type=str, required=False, default="max")
    parser.add_argument('--fix_constant_complexity',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--fix_negligeable_complexity',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temp_file_name_seed", type=str,
                        required=False, default="")
    parser.add_argument("--memory_limit", type=int,
                        required=False, default=None)
    parser.add_argument("--timeout", type=int, required=False, default=1)
    parser.add_argument("--large_timeout", type=int, required=False, default=1)
    parser.add_argument("--giga_timeout", type=int, required=False, default=30)
    parser.add_argument("--global_timeout", type=int,
                        required=False, default=1200)
    parser.add_argument("--max_workers", type=int,
                        required=False, default=1000000)

    parser.add_argument("--number_solutions_per_worker",
                        type=int, required=False, default=1)

    parser.add_argument("--main_process_cpu_id_list",
                        type=none_or_str, required=False, default='0,1')

    parser.add_argument("--forkserver_type", type=str,
                        required=False, default="standard")
    parser.add_argument('--use_distinct_forkservers',
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--forkserver_cpu_id_list",
                        type=none_or_str, required=False, default='2,3')
    parser.add_argument("--sandbox_cpu_id_step", type=int,
                        required=False, default=4)
    parser.add_argument("--sandbox_incremental_cpu_id_list",
                        type=none_or_str, required=False, default='0,1,2,3')
    parser.add_argument("--distinct_forkservers_incremental_cpu_id_list",
                        type=none_or_str, required=False, default='0,1,2,3')

    args = parser.parse_args()

    # now we need to know what part of the dataset will be handled by the job !

    # handle correctly the case of slurm_task_id (if at least one is None, falls back to the default folder_name)
    # temp_file_name_seed to update as well

    with open(Path(args.path_to_jsonl_file), 'r') as f:
        count_number_elements = sum(1 for _ in map(json.loads, f))

    print(f'{count_number_elements} elements in the dataset')

    # do some modifications if we are using some slurm array

    if (
        args.slurm_array_task_max is not None
    ) and (
        args.slurm_array_task_min is not None
    ) and (
        args.slurm_array_task_id is not None
    ) and (
        args.slurm_array_job_id is not None
    ):
        if args.code_end_index is None:
            args.code_end_index = count_number_elements

        print(f'Due to code_start_index and code_end_index value, in practice {args.code_end_index - args.code_start_index} elements in the dataset to study')

        print("Slurm job array, overwritting code_start_index, code_end_index, results_folder_name_root and temp_file_name_seed")
        number_of_jobs = (args.slurm_array_task_max -
                          args.slurm_array_task_min + 1)
        job_index = (args.slurm_array_task_id - args.slurm_array_task_min)

        job_index_list = np.array_split(
            list(range(args.code_start_index, args.code_end_index)), number_of_jobs)[job_index].tolist()
        args.code_start_index = min(job_index_list)
        args.code_end_index = max(job_index_list) + 1

        print(
            f'handling from {args.code_start_index} to {args.code_end_index}')

        args.temp_file_name_seed += f"_{args.slurm_array_job_id}_{args.slurm_array_task_id}"

    print("Logging parameters of the complexity framework")
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    framework_args = vars(args)

    run_complexity_framework(
        **framework_args
    )
