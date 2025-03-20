#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

set -eo pipefail

env_name=complexity

if ! command -v conda &> /dev/null; then
    echo "Conda not found! Please install Conda and make sure that it is available in your PATH." >&2

    exit 1
fi

echo "Creating '$env_name' Conda environment without torch..."

conda create\
    --yes\
    --name $env_name \
    --strict-channel-priority\
    --override-channels\
    --channel pytorch\
    --channel nvidia\
    --channel conda-forge \
    --channel defaults \
    python==3.10 \
    bubblewrap=0.8.0

echo "Installing more dependencies from pip..."

eval "$($(conda info --base)/bin/conda shell.bash hook)"
conda activate $env_name

# If you are using a different torch version, feel free to change the line below
pip install contourpy cycler fonttools joblib kiwisolver matplotlib \
    pyparsing python-dateutil scikit-learn scipy six threadpoolctl \
    exceptiongroup filelock iniconfig pluggy psutil pytest tomli

cat << EOF

Done!
To activate the environment, run 'conda activate $env_name'.
EOF
