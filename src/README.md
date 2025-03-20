<h1 align="center" style="font-variant: small-caps;">
    <p> 🛠️🧰 BigO(Bench) Framework 🧰🛠️ </p>
</h1>

## 👋 Overview

* [📋 Environment setup](#-environment-setup-back-to-top-back-to-root)

* [🔥1️⃣ Launch Inference Engine - `src/inference`](#1%EF%B8%8F⃣-launch-inference-engine---srcinference-back-to-top-back-to-root)

* [🔥2️⃣ Run Evaluation Inference - `src/eval`](#2%EF%B8%8F⃣-run-evaluation-inference---srceval-back-to-top-back-to-root)

* [🔥3️⃣ Post-process with the Complexity Framework - `src/complexity`](#3%EF%B8%8F⃣-post-process-with-the-complexity-framework---srccomplexity-back-to-top-back-to-root)

* [🔥4️⃣ Get the scores - `src/eval`](#4%EF%B8%8F⃣-get-the-scores---srceval-back-to-top-back-to-root)

* [🤹 Examples](#-examples-back-to-top-back-to-root)

* [License](#license-back-to-top-back-to-root)

* [📝 Citation](#-citation-back-to-top-back-to-root)

## 📋 Environment setup <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

If you don't need to install the repo-level dependency, and only want to work with one of the three modules [complexity](./complexity), [inference](./inference) and [eval](./eval), you can go check these folders and install the module-specific dependencies.

Otherwise, you can install once and for all the dependencies of the entire project with a single step script, that installs the environment [create_bigobench_env.sh](./create_bigobench_env.sh) !

```bash
cd src/
bash create_bigobench_env.sh
conda activate bigobench
```

In addition, if you want to test the evaluation on the data we made available on huggingface, be sure to download it first and put in the `../data` root folder. You can see more data instructions at [../data/README.md](../data/README.md).

## 🔥1️⃣ Launch Inference Engine - `src/inference` <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

As a start for running the BigO(Bench) evaluation, you will need an inference engine running your model. 
Three cases can happen:

<br>

* **You are using a HuggingFace model**:

    In this case, `src/inference` provides an inference engine based on VLLM.

    You can follow the set up steps at [`./inference/README.md`](./inference/README.md), that will allow you to launch such engine either in local or on a SLURM allocated machine (or even several engines at the same time using SLURM arrays !).

    This will create an engine running either at the address `0.0.0.0` if you are doing that locally, or at the address of a node `head_node` (or a comma separated list of nodes `node_1,node_2,node_3` if you are using SLURM arrays). The port will be 8000, as hard-coded in the scripts that follow.

    Remember this address at the `host` address. 

    That's all !

<br>

* **You are using an OpenAI model**:

    You only have a single step to do !

    Get your OpenAI API key (you will need to agree to the terms of use) at [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys).

    Then set the global environment variable `export OPENAI_API_KEY="your_openai_key_here"`.

    You won't need to set any host in what follows (discard this parameter in the commands), just remember your `model` which is the OpenAI API model name you will be calling.

<br>

* **You are using a custom engine**:
    
    BigO(Bench) can also handle custom inference engine.

    You will only need to make sure that it is compatible with the OpenAI Python API, that is to say that it can handle the same requests (completition, etc) at the same routes.

    If you custom engine is running at the local address `0.0.0.0` or any other address, that's what your host will be in the scripts that follow. Try to make your engine listen to port 8000 (default port for our scripts), otherwise you will need to change this hardcoded value in [./eval/eval.py](./eval/eval.py).

<br>

You are now settled with your LLM inference engine, you can then launch the evaluation inference. 

## 🔥2️⃣ Run Evaluation Inference - `src/eval` <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

Once you know which inference engine will run the LLM you are benchmarking, you can launch the evaluation script `./eval/eval.py` as detailed by the intructions of [`./eval/README.md`](./eval/README.md).

Using either the CLI directly or the SLURM scheduler, you will run a command of the type (again, more details on the commands are shared in [`./eval/README.md`](./eval/README.md)):

```bash 
cd eval
python -u eval.py \
    --host "host_address" \
    --model "model_name" \
    --max_concurrent_requests 256 \
    --task.data_file_path "../../data/time_complexity_test_set.jsonl" \
    --task.tasks_str "complexity_prediction/time_at_10,complexity_generation/time_at_10,complexity_ranking/time_at_10" \
    --task.write_eval "True" \
    --task.batch_size 256 \
    --task.use_sampling "True" \
    --task.temperature 0.8 \
    --task.top_p 0.95 \
    --dump_dir "./results"
```

> [!NOTE]
> If you are just testing the pipeline/you are in low-compute setting, you can replace `time_at_10` by `time_tiny_at_1`.
> This will modify the paths written below.

Where `host` is set as explained in the previous section (remove this argument if using the OpenAI API, after having set the `OPENAI_API_KEY`), `model` is the name of the model running in the inference engine (or the OpenAI API model name) and the rest of the arguments allow to run the evaluation for the time complexity tasks: 1. Time Complexity Prediction, 2. Time Complexity Generation, and 3. Time Complexity Ranking. Then, if you want the results on space complexity tasks, just change `time` into `space`.

The above command will create 3 output files storing the inference results on the 3 time complexity tasks:

* Time Complexity Prediction: `./eval/results/eval_results/complexity_prediction/time_at_10-time_complexity_test_set.jsonl`

* Time Complexity Generation: `./eval/results/eval_results/complexity_generation/time_at_10-time_complexity_test_set.jsonl`

* Time Complexity Ranking: `./eval/results/eval_results/complexity_ranking/time_at_10-time_complexity_test_set.jsonl`

(Note the folder `results` will be changed into whatever different you chose for the argument `--dump_dir`).

For the two tasks Complexity Generation and Complexity Ranking, you will need to post-process the above files with the Dynamic Complexity Inference Framework.

## 🔥3️⃣ Post-process with the Complexity Framework - `src/complexity` <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

Then, for both the outputs of the Complexity Generation and Complexity Ranking tasks, the complexity framework is used to post-process and get the complexity labels on the LLM-generated code snippets. Details are shared in [`./eval/README.md - 🍾 Complexity Generation Results`](./eval/README.md#-complexity-generation-results-back-to-top-back-to-root) and in [`./eval/README.md - 🎊 Complexity Ranking Results`](./eval/README.md#-complexity-ranking-results-back-to-top-back-to-root) on how to best connect the outputs of the previous step with the complexity framework. And of course, [`./complexity/README.md`](./complexity/README.md) gives extensive details on the way the complexity framework can be run.

Using the CLI or your SLURM scheduler, you will run on the above output files a command of the type:

```bash 
cd complexity
python -u -m command_line\
    --path_to_jsonl_file="../eval/results/eval_results/complexity_generation/time_at_10-time_complexity_test_set.jsonl"
```

where `--path_to_jsonl_file` specifies the path of the output file to be post-processed by the complexity framework. 

This will save the results in a .zip file of the type `src/complexity/results/results_datetime_xxx_id_yyy.zip`, if you are using the default value of the complexity framework argument `--results_folder_name_root` (if you are using SLURM arrays as described in the complexity framework documentation, the outputs will be saved in a folder `src/complexity/results/results_datetime_xxx_id_yyy/` filled with task_id zip files). Remember this path (a file for CLI or single SLURM, or a folder for SLURM arrays) as the source path for the post-processing.

Be it Complexity Prediction, Generation or Ranking, you are now ready to get all your scores :) 

## 🔥4️⃣ Get the scores - `src/eval` <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

Finally, using the post-processing scripts described in 
[`./eval/README.md - 3️⃣🔥 Post-processing of the metrics !`](./eval/README.md#3%EF%B8%8F⃣-post-processing-of-the-metrics--back-to-top-back-to-root), you can get the scores on all tasks of BigO(Bench):

<br>

* **Complexity Prediction**:

    Based on the output file of step "🔥2️⃣ Run Evaluation Inference", you can directly run the script that generates metrics for the complexity prediction task.

    ```bash
    cd eval
    python -u metrics/postprocessing_complexity_prediction.py \
        --results_file_path "results/eval_results/complexity_prediction/time_at_10-time_complexity_test_set.jsonl" \
        --dump_dir "./results" \
        --test_set_file_path "../../data/time_complexity_test_set.jsonl" \
        --time_or_space "time" \
        --generate_baseline False \
        --at_k 10
    ```

    This will print scores for the task in the CLI and save them in `./eval/results/postprocessed_results.json`.

<br>

* **Complexity Generation**:

    Based on the output file of step "🔥3️⃣ Post-process with the Complexity Framework", you can directly run the script that generates metrics for the complexity generation task.

    ```bash
    cd eval
    python -u metrics/postprocessing_complexity_generation.py \
        --results_folder_or_file_path "../complexity/results/results_datetime_xxx_id_yyy.zip" \
        --dump_dir "./results" \
        --test_set_file_path "../../data/time_complexity_test_set.jsonl" \
        --time_or_space "time" \
        --generate_baseline False \
        --at_k 10 
    ```

    `--results_folder_or_file_path` is either a .zip file path (if you ran the complexity framework with the CLI or single SLURM) or a folder path (containing .zip files, if you used SLURM array).

    This will print scores for the task in the CLI and save them in `./eval/results/postprocessed_results.json`.

<br>

* **Complexity Ranking**:

    Based on the output file of step "🔥3️⃣ Post-process with the Complexity Framework", you can directly run the script that generates metrics for the complexity ranking task.

    ```bash
    cd eval
    python -u metrics/postprocessing_complexity_ranking.py \
        --results_folder_or_file_path "../complexity/results/results_datetime_xxx_id_yyy.zip" \
        --dump_dir "./results" \
        --test_set_file_path "../../data/time_complexity_test_set.jsonl" \
        --time_or_space "time" \
        --generate_baseline False \
        --at_k 10 
    ```

    `--results_folder_or_file_path` is either a .zip file path (if you ran the complexity framework with the CLI or single SLURM) or a folder path (containing .zip files, if you used SLURM array).

    This will print scores for the task in the CLI and save them in `./eval/results/postprocessed_results.json`.

<br>

You now have all your scores saved in `./eval/results/postprocessed_results.json` (except if you changed `--dump_dir`, which will save results in your own specified folder). 

Feel free to contact us if you would like to include them in our [🏆 Leaderboard 🏆](https://facebookresearch.github.io/bigobench/leaderboard.html).

## 🤹 Examples <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

Finally, to illustrate all the explanations above, we provide 4 step-by-step detailed document to run BigO(Bench) in the following configurations:

* [`./example_huggingface_with_slurm.md`](./example_huggingface_with_slurm.md): you want to benchmark a HuggingFace model, and you have a cluster of machines with SLURM.

* [`./example_openai_with_slurm.md`](./example_openai_with_slurm.md): you want to benchmark an OpenAI API model, and you have a cluster of machines with SLURM.

* [`./example_huggingface_in_local.md`](./example_huggingface_in_local.md): you want to benchmark a HuggingFace model, and you only have a local machine (this is not the preferred configuration, you might lack GPU-CPU resources).

* [`./example_openai_in_local.md`](./example_openai_in_local.md): you want to benchmark an OpenAI API model, and you only have a local machine (this is not the preferred configuration, you might lack CPU resources).

## License <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

The majority of BigO(Bench) is licensed under CC-BY-NC (see [LICENCE](/LICENSE.md)), however portions of the project are available under separate license terms: https://github.com/pberkes/big_O is licensed under the BSD-3 license.

## 📝 Citation <sub><sup>([back to top](#-overview)) ([back to root](../README.md#-project-overview-back-to-top))<sub><sup>

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