#!/bin/bash

## Copyright (c) Meta Platforms, Inc. and affiliates.
## All rights reserved.

## This source code is licensed under the license found in the
## LICENSE file in the root directory of this source tree.

#SBATCH --nodes=1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task ***TODO***
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/complexity-%x.%A.%a.%j.out
#***TODO*** set any addition parameters for SBATCH (account, partition, ...)
#SBATCH --time=1:00:00
#SBATCH --exclusive
#SBATCH --mem 0
#SBATCH --array=1-1%1

python -u -m command_line\
    --path_to_jsonl_file="source_of_data.jsonl"\
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID\
    --slurm_array_task_max=$SLURM_ARRAY_TASK_MAX\
    --slurm_array_task_min=$SLURM_ARRAY_TASK_MIN\
    --slurm_array_job_id=$SLURM_ARRAY_JOB_ID