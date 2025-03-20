#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

env_name=eval

if ! command -v conda &> /dev/null; then
    echo "Conda not found! Please install Conda and make sure that it is available in your PATH." >&2

    exit 1
fi

echo "Creating '$env_name' Conda environment with torch and other dependencies..."

conda create\
    --yes\
    --name $env_name \
    --strict-channel-priority\
    --override-channels\
    --channel pytorch\
    --channel nvidia\
    --channel conda-forge \
    --channel defaults \
    python==3.10.14 \
    bubblewrap=0.8.0


echo "Installing vllm from pip..."

eval "$($(conda info --base)/bin/conda shell.bash hook)"
conda activate $env_name

# If you are using a different torch version, feel free to change the line below
pip install torch 

pip install numpy typeguard tenacity openai httpx tree_sitter 

cat << EOF

Done!
To activate the environment, run 'conda activate $env_name'.
EOF
