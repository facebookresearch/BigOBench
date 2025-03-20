## Copyright (c) Meta Platforms, Inc. and affiliates.
## All rights reserved.

## This source code is licensed under the license found in the
## LICENSE file in the root directory of this source tree.

#SBATCH --nodes=1
#SBATCH --tasks-per-node 1
#SBATCH --cpus-per-task ***TODO***
#SBATCH --gpus-per-node=0
#SBATCH --output=logs/eval-%x.%A.%a.%j.out
#***TODO*** set any addition parameters for SBATCH (account, partition, ...)
#SBATCH --time=10:00:00
#SBATCH --exclusive
#SBATCH --mem 0
#SBATCH --array=1-1%1

python -u eval.py \
    --host "host_address" \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --max_concurrent_requests 256 \
    --task.data_file_path "../../data/time_complexity_test_set.jsonl" \
    --task.tasks_str "complexity_prediction/time_at_10,complexity_generation/time_at_10" \
    --task.write_eval "True" \
    --task.batch_size 256 \
    --task.use_sampling "True" \
    --task.temperature 0.8 \
    --task.top_p 0.95 \
    --dump_dir "./results"