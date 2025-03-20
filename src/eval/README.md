<h1 align="center" style="font-variant: small-caps;">
    <p>⚖️✏️ BigO(Bench) Evaluation ⚖️✏️</p>
</h1>

Here we detail how to run evaluation of any models from Huggingface or directly the OpenAI interface. These scripts also work as long as you have a custom setup serving the routes used by the openAI API at a specific address. Our evaluation includes the following three tasks:

* The first evaluation task of the benchmark, `Complexity Prediction`, consists in predicting the time and space complexity given a problem description and a human solution. Our baseline for this task is the naive model that always returns O(n), the most frequent class. Pass@k measures the accuracy of finding the correct complexity; Best@k measures accuracy only across the most optimized complexity class of each problem; All@k requires correct complexity output across all complexity classes at once per problem.

* The second task `Complexity Generation` requires the LLM to generate a correct solution to a given problem description that has to respect a feasible time or space complexity requirement. Our baseline for this task is a Llama 3.1 70B model that is queried for the same prompts without the complexity requirement. Pass@k measures the accuracy of finding a correct solution, according to public, private and generated tests, that has the correct complexity, as measured by the complexity framework; Best@k and All@k are similarly defined as their counterparts in the results of the first task.

* The third task, `Complexity Coefficient Percentile Ranking`, measures how a generated solution to a given problem, respecting a complexity requirement, ranks among human solutions of the same complexity class and problem. The ranking is performed based on the coefficient of the complexity curve, as measured by the framework: the lower the coefficient, the more flat the complexity curve and the more optimized the solution. Ranking results are given in percentile of the distribution, where a solution of the nth percentile is more optimized than n% of human solutions. The querying is similar to the second task with the addition of the requirement "Try to optimize the runtime of your code as much as you can, while respecting the time complexity requirement".

## 👋 Overview 

* [📋 Environment setup](#-environment-setup-back-to-top-back-to-root)

* [1️⃣🔥 Inference Engine](#1%EF%B8%8F%E2%83%A3-inference-engine-back-to-top-back-to-root)

    * [🧑‍💻 Using OpenAI](#-using-openai-back-to-top-back-to-root)

    * [🤖 Using VLLM](#-using-vllm-back-to-top-back-to-root)

    * [Using anything else](#using-anything-else-back-to-top-back-to-root)

* [2️⃣🔥 Launching evaluation inference !](#2%EF%B8%8F⃣-launching-evaluation-inference--back-to-top-back-to-root)

    * [🔨 CLI entry point](#-cli-entry-point-back-to-top-back-to-root)

    * [🛠️ SLURM entry point !](#%EF%B8%8F-slurm-entry-point--back-to-top-back-to-root)

    * [📚 The different tasks !](#-the-different-tasks--back-to-top-back-to-root)

    * [⚙️ More evaluation parameters](#%EF%B8%8F-more-evaluation-parameters-back-to-top-back-to-root)

    * [📑 Outputs of the evaluation inference](#-outputs-of-the-evaluation-inference-back-to-top-back-to-root)

* [3️⃣🔥 Post-processing of the metrics !](#3%EF%B8%8F⃣-post-processing-of-the-metrics--back-to-top-back-to-root)

    * [🎂 Complexity Prediction Results](#-complexity-prediction-results-back-to-top-back-to-root)

    * [🍾 Complexity Generation Results](#-complexity-generation-results-back-to-top-back-to-root)

    * [🎊 Complexity Ranking Results](#-complexity-ranking-results-back-to-top-back-to-root)

    * [🎈 Results outputs](#-results-outputs-back-to-top-back-to-root)

* [License](#license-back-to-top-back-to-root)

* [📝 Citation](#-citation-back-to-top-back-to-root)

## 📋 Environment setup <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

If you already have installed the repo-level dependencies that's great, you've got nothing to do !

In case you would like to run the evaluation only, without installing the other dependencies of this repo, we provide the following instructions to install the dependencies.

[create_eval_env.sh](./create_eval_env.sh) is a single step script that installs the environment.

```bash
cd src/eval
bash create_eval_env.sh
conda activate eval
```

In addition, if you want to test the evaluation on the data we made available on huggingface, be sure to download it first and put in the `data` root folder. You can see more data instructions at [README.md](../../data/README.md).

## 1️⃣🔥 Inference Engine <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

First, you'll need to have an inference engine set up in order to run the evaluation on BigO(Bench). The evaluation scripts provided in this module rely on the python `openai` module, therefore any engine that uses these routes can be used for our evaluations.

### 🧑‍💻 Using OpenAI <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

In order to use OpenAI engine (that is to say their online API), you just need to set your openAI key as an environment variable.
```bash
export OPENAI_API_KEY="your_openai_key_here"
```

Then, in the scripts that follow, do no set the `--host` parameter (or set it to the empty string). The evaluation script will understand that in the absence of a specified host, it should default to querying the OpenAI API. 
Do not forget to still set the `--model` parameter to the model you want to use in OpenAI, as detailed in their API documentation. In addition, for the O1 serie of models, we do provide a flag `--light_request_arguments`, as the arguments to pass to the request for such models are slightly different than for other OpenAI models.

### 🤖 Using VLLM <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

Otherwise, you can use your own Huggingface model checkpoitns with VLLM, that usually support most commonly used models uploaded on this platform. For this, you will need to launch a VLLM server first, using the scripts and the documentation provided in [/src/inference](/src/inference). Once such a server is running, it will expose a host that you can query using the OpenAI Python API.

To use the running VLLM engine, write down the address of the host, which can be `0.0.0.0` if you are running everything with the CLI in local, or `head_node` if launching VLLM on a SLURM node. Then, in the scripts that follows, set to the parameter `--host` to the host name as explained just before. The evaluation script will understand that in the presence of a specified host, it should query this host in particular. If you are encountering issues to reach the VLLM instance, it might be due to the port that is exposed (our scripts use the port 8000, as defined within the script `/src/eval/eval.py`).

Do not forget to specify the model name, as served by the VLLM engine, using `--model`.

If you need to change any of the other hard-coded parameters, you can have a deeper look at the file [/src/eval/eval.py](/src/eval/eval.py), where all the routines to query the models is located. For instance, timeout values are defined as follows:

```bash
OpenAI(
    base_url=url, 
    api_key="EMPTY",
    max_retries = 1,
    timeout = httpx.Timeout(timeout=5000.0, connect=5.0),
) 
```

### Using anything else <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

You can use any other inference engine that is compatible with the OpenAI Python module, that is to say serving the same routes as the OpenAI API. You can leverage the `--host` parameter to set the address where the model is running, and `--model` to set the model name. You might need to change the port value, hard-coded in the script, to a different value (currently set to 8000).

## 2️⃣🔥 Launching evaluation inference ! <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

Once your inference engine is defined, be it through VLLM, the OpenAI API or any custom inference engine, you are ready to launch the evaluation inference !
You can use several entry points to run the tasks of BigO(Bench). For the two tasks of Complexity Generation and Complexity Ranking, you'll need to run the dynamic complexity inference framework on top of the generated answers of the model, in order to evaluate their correctness with regards to time and space complexity. For all the tasks, you'll finally have to use a post-processing script that outputs the scores as presented in our paper and website !

### 🔨 CLI entry point <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

You can directly run the evaluation inference, using the correct `--host` and `--model` parameters as explained above in [Inference Engine](## 1️⃣🔥 Inference Engine).
In all that follows, if you want to test your setting, replace `time_at_10` (present two times in the argument `--task.tasks_str`) by `time_tiny_at_1`.

```bash
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
```

So following the previously shared details on how to use the OpenAI API, if you are using GPT4o through their API, just do


```bash
export OPENAI_API_KEY="your_openai_key_here"

python -u eval.py \
    --model "gpt-4o" \
    --max_concurrent_requests 256 \
    --task.data_file_path "../../data/time_complexity_test_set.jsonl" \
    --task.tasks_str "complexity_prediction/time_at_10,complexity_generation/time_at_10" \
    --task.write_eval "True" \
    --task.batch_size 256 \
    --task.use_sampling "True" \
    --task.temperature 0.8 \
    --task.top_p 0.95 \
    --dump_dir "./results"
```

On the contrary, if you are using a SLURM-based VLLM instance, running on node `node-1`, just do

```bash
export OPENAI_API_KEY="your_openai_key_here"

python -u eval.py \
    --host "node-1" \
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
```

### 🛠️ SLURM entry point ! <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

You can also launch the CLI command with slurm, so to have a dedicated node orchestrating the evaluation inference.

```bash
sbatch slurm.sh
```

As detailed in the section before, be sure to set `--host` and `--model` so to use either the OpenAI API, your own VLLM instance(s) or a custom inference engine, compatible with the OpenAI Python API.

### 📚 The different tasks ! <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

Whenever you want to add a task to the evaluation inference, just add them to the argument `--task.tasks_str` and separate each task with a comma. Evaluation tasks all follow the following formatting:

```bash
{task}/{subtask}_{tiny_or_not}_{at_k}
```

We detail each component of the task names:

* `task`:

    * `complexity_prediction`, that consists in asking the LLM the complexity of a human-generated piece of code. This complexity, predicted by the LLM, is then compared with the ground truth complexity as predicted by the dynamic complexity inference framework when run on the human-generated piece of code.

    * `complexity_generation`. The LLM is asked to generate a piece of code that solves the coding challenge, while respecting a complexity requirement. The complexity requirement is supposed to be achievable, as determined by the complexity framework when run on all human ground-truth solutions on the specific question.

    * `complexity_ranking`, in which case the LLM solves a similar task to the previous `complexity_generation` task, with a slightly modified prompt, except that on top of checking the complexity, the LLM-generated solution is then ranked among the human solutions in terms of complexity curve coefficient, using percentile ranking.

* `subtask`:

    * `time`, the standard time-complexity of each of the above tasks. For instance, `complexity_prediction/time...` consists in asking the LLM to predict the time complexity of the human piece of code.

    * `space`, the space-complexity variant of the above tasks.

    * `time_more_detailed_instructions`. This subtask is specific to `complexity_prediction`. It allows for a more detailed instruction query for the LLM, supposed to remove some ambiguity from the question. For the complexity prediction scores reported in the paper, we used this subtask (`time_more_detailed_instructions`). But it would be more natural for a human to use the standard `time`, that includes more ambiguity in the definition of what the complexity is.

    * `space_more_detailed_instructions`. The space variant of the above `time_more_detailed_instructions` subtask. This is the subtask used to the space complexity prediction scores reported in the paper. This subtask is also only available for `complexity_prediction`.

    * `no_conditioning`. This subtask is specific to `complexity_generation`, and enables to ask for the LLM to generate a piece of code solving the problem, without giving any complexity requirement, be it for time complexity or space complexity. This is what the paper uses as one of the baselines, to control that the complexity requirement brings indeed some improvements of the LLM scores.

* `tiny_or_not` is used to restrict the test sets to a small fraction, mainly for debugging purposes. Set this value to `tiny` if you would like to use this setting, otherwise skip it (e.g. `complexity_prediction/time_at_10`).

* `at_k` specifies the value of pass@k. When set to 1, the LLMs will be queried only one time per test case. For higher values, they will be queried 2*k times, so to approximate correctly the value of pass@k.

### ⚙️ More evaluation parameters <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

The script `eval.py` can be conditioned for various parameters, as detailed below:

* `--host` which is the address hosting the inference engine. If using VLLM, set to the local address serving the model, for instance `0.0.0.0`. If serving from a different SLURM node, set to the head node name, `node-1`. If using OpenAI, do no set (this will default this argument to None).

* `--model` is the name of the model being queried, be it the name of the model deployed by the VLLM instance like `meta-llama/Llama-3.1-70B-Instruct` or the name of a model served by the OpenAI API like `gpt-4o`.

* `--max_concurrent_requests` is the maximum number of concurrent requests (sending questions to the LLM inference engine) at a time. This has to be trade-off with the limits of the main machine running the eval harness, as each concurrent request uses a separate thread (you might want to limit the number of queries being filed in parallel) to send the request and analyse the returned answer. This is also a trade-off with the LLM engine itself, that might have rate limits. For OpenAI, we usually use a value of `256`, but if you start seeing a lot of errors in the logs (probably due to rate limits), feel free to lower this value. When using one VLLM engine with good memory (so that can store a lot of incoming requests), you can use higher values than `256`, especially if you are using our slurm array jobs that allow to launch several concurrent VLLM engines. With 50 engines and a good main machine to run the eval harness, you can easily set this value to `16384`.

* `--light_request_arguments` is an optionnal argument used for certain models of the OpenAI API, like o1, that do not use the same arguments as the regular OpenAI Python API. If you set `--model o1-mini` for instance, then use the flag `--light_request_arguments "True"` as well.

* `--max_tokens` defines the maximal number of tokens to be generated by the inference engine. To let token space reasoning models freely use their 'think' tokens, we set this value to `--max_tokens 16384` during all of our evaluations (all models had at least 32k context window).

* `--dump_dir` the directory to store the evaluation outputs. By default this will be `./results`

* `--task`, various sub-arguments specific to the inference engine itself:

    * `--task.data_file_path` is the path to the file that contains the data to evaluate on, including the ground truth complexity labels. These should be in the same format as our time and space test sets, shared on HuggingFace. If you have followed the instructions to download them (see [data](../../data)), you can set `--task.data_file_path "../../data/time_complexity_test_set.jsonl"` for the time complexity based sub-tasks (otherwise change to the space test set file).

    * `--task.tasks_str` is a comma separated list of task names, as explained above in ["📚 The different tasks !"](#-the-different-tasks-). If running on `--task.data_file_path "../../data/time_complexity_test_set.jsonl"`, you can do all the time-complexity tasks at once with pass@10, using `--task.tasks_str "complexity_prediction/time_at_10,complexity_generation/time_at_10,"complexity_ranking/time_at_10"`.

    * `--task.write_eval` set to `--task.write_eval "True"` if you would like to save the evaluations.

    * `--task.batch_size` batch size to run on.

    * `--task.use_sampling` if set to `--task.use_sampling "True"`, will use temperature sampling, otherwise will make temperature equals to zero.

    * `--task.temperature` used for temperature sampling. We used a temperature of 0.8 and 0.95 top_p for all models, except deepseek r1 for which we used temperature 0.6 as advised.

    * `--task.top_p` used for nucleus sampling, usually 0.95.

### 📑 Outputs of the evaluation inference <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

If using `--task.write_eval "True"` in the above evaluation commands, results will be saved to `--dump_dir`, which by defaults is in `./results`. In this folder, you will find:

* `eval_params.json` that records all the evaluation parameters you used to launch the evaluation inference.

* `temp_results.json` that gives temporary metrics, such as non-post-processed complexity prediction metrics, pass@k for program correctness (but not for complexity generation), also non post-processed, ... This can be used as a proxy of the final metrics you would get after post-processing. But keep in mind that these results are biased, for instance they do not account for imbalance of the data in the test sets. And on top of it they are only partial for the complexity generation and ranking tasks, that require to run the complexity framework before being able to output any concrete results.

* `complexity_prediction/`, `complexity_generation/` and `complexity_ranking/`. These folders, created if you indeed run the corresponding evaluation inference, will contain a .jsonl storing the details of each evaluation inference on each test set sample. These are the files that will be fed to the post-processing scripts. Notice that in these files, any original field from the test sets named `time_complexity_inferred` or `space_complexity_inferred`, which corresponds to the "ground-truth" labels of the human solutions as inferred by the complexity framework, will be renamed in the jsonl evaluation outputs into `time_complexity_synthetic_ground_truth` and `space_complexity_synthetic_ground_truth`. This is to distinguish these fields from any inferred label by the complexity framework on the LLM generated solutions, that will be synthetic labels to be compared with the synthetic ground truth labels.

## 3️⃣🔥 Post-processing of the metrics ! <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

Last step to get your model results in BigOBench ! 🥳

After running the evaluation inference of the step above, you'll have a post-processing to do for each task, in order to get the row of results exactly as shared on our Website and in our paper.

### 🎂 Complexity Prediction Results <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

Make sure you used the flag `--task.write_eval "True"` during the evaluation inference, so that the model outputs are saved. Using the specified output folder with `--dump_dir` (default is `src/eval/results/`), run the below command on the file of outputs to get your results:

```bash
python -u metrics/postprocessing_complexity_prediction.py \
    --results_file_path "results/eval_results/complexity_prediction/time_at_10-time_complexity_test_set.jsonl" \
    --dump_dir "./results" \
    --test_set_file_path "../../data/time_complexity_test_set.jsonl" \
    --time_or_space "time" \
    --generate_baseline False \
    --at_k 10
```

The command above will work great if you did the time complexity prediction task, using the task string `complexity_prediction/time_at_10`.
If you instead did space complexity, please use:

```bash
python -u metrics/postprocessing_complexity_prediction.py \
    --results_file_path "results/eval_results/complexity_prediction/space_at_10-space_complexity_test_set.jsonl" \
    --dump_dir "./results" \
    --test_set_file_path "../../data/space_complexity_test_set.jsonl" \
    --time_or_space "space" \
    --generate_baseline False \
    --at_k 10
```

This post-processing will allow to perform aggregation accross test set samples so to minimize imbalance, and report the most realistic results. 

### 🍾 Complexity Generation Results <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

For complexity generation and ranking tasks, you will need to run the complexity framework first, and then a script similar to the one above for the complexity prediction task. The reason why the complexity framework run is not groupped with the post-processing within a single script, and why the complexity framework is also separate from the sandboxes used to check for program correctness during the evaluation inference script, is that the complexity framework can require way more compute than the previous steps and is more subject to process noise, if running concurrently with too many other processes, used for LLM inference for instance (if your LLM inference engined is deployed locally on the CPU cores as the ones where the complexity framework is running). This is way we made the evaluation pipeline run the complexity framework separately.

The framework can be run directly on the .jsonl file output from the evaluation inference, for instance with the default value of `--dump_dir` located at `results/eval_results/complexity_generation/time_at_10-time_complexity_test_set.jsonl`. Just change the data file path in the complexity framework scripts to this output file, and you'll have the complexity framework run and output what is needed for the postprocessing.

```bash
cd ../complexity
sbatch slurm.sh
```

Where `slurm.sh` can be modified to

```bash
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
    --path_to_jsonl_file="../eval/results/eval_results/complexity_generation/time_at_10-time_complexity_test_set.jsonl"\
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID\
    --slurm_array_task_max=$SLURM_ARRAY_TASK_MAX\
    --slurm_array_task_min=$SLURM_ARRAY_TASK_MIN\
    --slurm_array_job_id=$SLURM_ARRAY_JOB_ID
```

The complexity framework will use its own argument `--results_folder_name_root` to know which folder to use for its outputs, by default it will be `src/complexity/results/results_datetime_xxx_id_yyy.zip` if you are using the command_line/simple slurm to run the complexity framework, or `src/complexity/results/results_datetime_xxx_id_yyy/` filled with task_id zip files if running the complexity framework with slurm array.

Then, using either the zip file path (Slurm/CLI) or the folder path (Slurm array), you can input the results of the complexity framework run on the outputs of the LLM on the evaluation to get the generation outputs.

```bash
python -u metrics/postprocessing_complexity_generation.py \
    --results_folder_or_file_path "../complexity/results/results_datetime_xxx_id_yyy.zip" \
    --dump_dir "./results" \
    --test_set_file_path "../../data/time_complexity_test_set.jsonl" \
    --time_or_space "time" \
    --generate_baseline False \
    --at_k 10 
```

And you are good to go ! 


### 🎊 Complexity Ranking Results <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

This section is very similar to the details shared above for the complexity generation results. We will just adapt the code examples for the ranking task (though the modifications to be done are as simple as `replace("generation", "ranking")`).

```bash
cd ../complexity
sbatch slurm.sh
```

Where `slurm.sh` can be modified to

```bash
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
    --path_to_jsonl_file="../eval/results/eval_results/complexity_ranking/time_at_10-time_complexity_test_set.jsonl"\
    --slurm_array_task_id=$SLURM_ARRAY_TASK_ID\
    --slurm_array_task_max=$SLURM_ARRAY_TASK_MAX\
    --slurm_array_task_min=$SLURM_ARRAY_TASK_MIN\
    --slurm_array_job_id=$SLURM_ARRAY_JOB_ID
```

And then:

```bash
python -u metrics/postprocessing_complexity_ranking.py \
    --results_folder_or_file_path "../complexity/results/results_datetime_xxx_id_yyy.zip" \
    --dump_dir "./results" \
    --test_set_file_path "../../data/time_complexity_test_set.jsonl" \
    --time_or_space "time" \
    --generate_baseline False \
    --at_k 10 
```

And you are good to go ! 

### 🎈 Results outputs <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>


## License <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

The majority of BigO(Bench) is licensed under CC-BY-NC (see [LICENCE](/LICENSE.md)), however portions of the project are available under separate license terms: https://github.com/pberkes/big_O is licensed under the BSD-3 license.

## 📝 Citation <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>
If you find our project useful and/or are using its data, please cite our paper:
```
@misc{chambon2025bigobenchllmsgenerate,
      title={BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?}, 
      author={Pierre Chambon and Baptiste Roziere and Benoit Sagot and Gabriel Synnaeve},
      year={2025},
      eprint={2503.15242},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.15242}, 
}
```