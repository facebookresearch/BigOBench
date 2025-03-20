#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

env_name=vllm

if ! command -v conda &> /dev/null; then
    echo "Conda not found! Please install Conda and make sure that it is available in your PATH." >&2

    exit 1
fi

echo "Creating '$env_name' Conda environment with torch and other dependencies bundled with vllm..."

conda create\
    --yes\
    --name $env_name \
    --strict-channel-priority\
    --override-channels\
    --channel pytorch\
    --channel nvidia\
    --channel conda-forge \
    python==3.10.14

echo "Installing vllm from pip..."

eval "$($(conda info --base)/bin/conda shell.bash hook)"
conda activate $env_name

pip install vllm ipython fire

cat << EOF

Done!
To activate the environment, run 'conda activate $env_name'.
EOF
