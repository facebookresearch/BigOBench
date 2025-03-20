<h1 align="center">
    <p>🏎️ Inference with VLLM 🚄</p>
</h1>

In order to perform evaluation on any HuggingFace open-sourced model, we use VLLM that allows to perform inference at scale on any weights. For some very recently released models, you might need to wait for them to push an updated version, and then update your vllm python package for the model to be ready to use. VLLM offers an interface pretty much similar to the one of OpenAI API. VLLM consists in starting a model server on a bunch of nodes. This server can then receive API requests for model inference.

## 👋 Overview 

* [📋 Environment setup](#-environment-setup-back-to-top-back-to-root)

* [🔥 Launching vllm server - in local ! ](#-launching-vllm-server---in-local--back-to-top-back-to-root)

* [🔥🔥 Launching vllm server - using SLURM !](#-launching-vllm-server---using-slurm--back-to-top-back-to-root)

* [🧪🔬 Quick VLLM inference](#-quick-vllm-inference-back-to-top-back-to-root)

* [License](#license-back-to-top-back-to-root)

* [📝 Citation](#-citation-back-to-top-back-to-root)

## 📋 Environment setup <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

In case you would like to run the inference with VLLM only, without installing the other dependencies of this repo, we provide the following instructions to install the dependencies.

[create_vllm_env.sh](./create_vllm_env.sh) is a single step script that installs the environment.

```bash
bash create_vllm_env.sh
conda activate vllm
```

## 🔥 Launching vllm server - in local ! <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

You can run the below command to launch a VLLM server in local. Though certain models will fit on a single GPU, we usually use 8 for the model `meta-llama/Llama-3.1-70B-Instruct`. You can specify the number of GPUs on which to parallelize using `--tensor-parallel-size=$NUM_GPUS`.

We also advise to download the model weghts before, so to avoid having concurrent downloads behind launched in parallel by different VLLM instances, which could create some conflicts in the downloaded weights.
meta-llama models usually need approval to be used, by going on the Huggingface website and accepting the terms of use on the model page first. Then be sure to download weights in local after logging-in with the huggingface CLI tool, using the same login as the one that accepted the terms of use of the model online.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct
```

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct --enforce-eager --max-model-len=32000 --tensor-parallel-size=8  --trust-remote-code
```

This will start a server at the address `http://0.0.0.0:8000`.

## 🔥🔥 Launching vllm server - using SLURM ! <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

Make sure to be in a conda environment where vllm is available (using the environment setup above, or the global env for the repo in `/src`).
You can then launch a vllm server of the model you wish to use.

We also advise to download the model weghts before, so to avoid having concurrent downloads behind launched in parallel by different VLLM instances, which could create some conflicts in the downloaded weights.

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct
```

```bash
sbatch start_vllm_server_llama_70B.sh
```

Be sure to edit `start_vllm_server_llama_70B.sh` so to specify the SLURM headers of your configuration.

If you wish to use different models, you just need to touch 3 lines at the start of the file `start_vllm_server_llama_70B.sh` to change set up

```bash
# tune parameters below
MODEL=meta-llama/Llama-3.1-405B-Instruct
MAX_MODEL_LEN=32000
GPU_UTILIZATION=0.9
```

You might want to change the number of nodes in the file sbatch headers for particularly long context windows.

If you want to run inference on the DeepSeekR1-llama models, you would need to use

```bash
# tune parameters below
MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
MAX_MODEL_LEN=32000
GPU_UTILIZATION=0.9
```

In addition, notice that the script that we provide is compatible with slurm arrays, if you want to launch several instances in parallel. 
In which case you can tune the slurm array parameters at the top

```bash
#SBATCH --array=1-n%k
```

Be sure in this case to have `n` and `k` set to the same value, for instance 10, if you wish to have 10 instances running concurrently.

Within 5-10mn after your job started running, you should have your server ready. Check the slurm.out file to correctly display at the end
```bash
INFO:     Application startup complete.
```

This means that a vllm instance is accessible at the address `http://head_node:8000` where `head_node` is the slurm node ID of one of all the nodes on which this instance is running. If you are running `squeue`, you should see with `NODELIST` corresponding to your job a list `node-1,node-2` if your VLLM server is running on these two nodes. In which case you can set `head_node=node-1` in the address above (which becomes `http://node-1:8000`).

For convenience, we do provide a script that lists the head-nodes of all the instances running, especially when you are using a slurm array.
If you launched `start_vllm_server_llama_70B.sh` with the slurm array parameters `#SBATCH --array=1-64%64`, you can run:

```bash
bash get_list_of_head_nodes.sh start_vllm_server_llama_70B.sh
```

which will give you a comma-separated list of head_nodes on which these instances are running, for at least 10mn (which is usually the start up time of a VLLM instance).

This will output

```bash
Nodes of running jobs for user : node-1,node-3,node-5,node-7,node-9,node-11,node-13,node-15,node-17,node-19,node-21,node-23,node-25,node-27,node-29,node-31,node-33,node-35,node-37,node-39,node-41,node-43,node-45,node-47,node-49,node-51,node-53,node-55,node-57,node-59,node-61,node-63,node-65,node-67,node-69,node-71,node-73,node-75,node-77,node-79,node-81,node-83,node-85,node-87,node-89,node-91,node-93,node-95,node-97,node-99,node-101,node-103,node-105,node-107,node-109,node-111,node-113,node-115,node-117,node-119,node-121,node-123,node-125,node-127
```

In addition, we also provide a script that can check the sanity of the above instances, in case some are failing. After running the above command `bash get_list_of_head_nodes.sh start_vllm_server_llama_70B.sh`, you can run:

```bash
python give_out_healthy_vllm.py\
  --host_list node-1\
  --model meta-llama/Llama-3.1-70B-Instruct\
  --timeout 240
```

Or with multiple nodes:

```bash
python give_out_healthy_vllm.py\
  --host_list node-1,node-3,node-5,node-7,node-9,node-11,node-13,node-15,node-17,node-19,node-21,node-23,node-25,node-27,node-29,node-31,node-33,node-35,node-37,node-39,node-41,node-43,node-45,node-47,node-49,node-51,node-53,node-55,node-57,node-59,node-61,node-63,node-65,node-67,node-69,node-71,node-73,node-75,node-77,node-79,node-81,node-83,node-85,node-87,node-89,node-91,node-93,node-95,node-97,node-99,node-101,node-103,node-105,node-107,node-109,node-111,node-113,node-115,node-117,node-119,node-121,node-123,node-125,node-127\
  --model meta-llama/Llama-3.1-70B-Instruct\
  --timeout 240
```

This will output a comma-separated list of head_nodes which are also answering dummy requests:


## 🧪🔬 Quick VLLM inference <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

If you want to test your set up, you can use the script `test_vllm.py`.
To test a local deployment of vllm, just run:

```bash
python test_vllm.py --host 0.0.0.0 --model meta-llama/Llama-3.1-70B-Instruct
```

To test one of the instances of a SLURM deployment of VLLM, just use one of the head_node list:

```bash
python test_vllm.py --host head_node --model meta-llama/Llama-3.1-70B-Instruct
```


## License <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>

The majority of BigO(Bench) is licensed under CC-BY-NC (see [LICENCE](/LICENSE.md)), however portions of the project are available under separate license terms: https://github.com/pberkes/big_O is licensed under the BSD-3 license.

## 📝 Citation <sub><sup>([back to top](#-overview)) ([back to root](../../README.md#-project-overview-back-to-top))<sub><sup>
If you find this repository useful, please cite this as
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