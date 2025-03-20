#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

if [ $# -eq 0 ]; then
    echo "Error: Please provide a job name"
    exit 1
fi
# Get the username from the command line argument
JOBNAME=$1

# Get the list of running jobs
RUNNING_JOBS=$(squeue -u "$USER" -n "$JOBNAME" -h -t RUNNING -o "%M %N" | grep -E '^[1-9]-[0-9]{2}:[0-9]{2}:[0-9]{2}|^[1-9]:[0-9]{2}:[0-9]{2}|^[1-9][0-9]:[0-9]{2}' | awk '{print $NF}')


# # Compile the list of nodes into a comma-separated string
# NODES=$(echo "$RUNNING_JOBS" | | tr '\n' ',' | sed 's/,$//')

# echo "Nodes of running jobs: $NODES"

# Process each element of $RUNNING_JOBS
PROCESSED_NODES=()
while IFS= read -r node; do
    # Remove brackets and extract the first node
    node=${node#[}
    node=${node%]}
    node=${node%%,*}
    node=${node//[/}
    node=${node//]/}
    PROCESSED_NODES+=("$node")
done <<< "$RUNNING_JOBS"
# Compile the list of nodes into a comma-separated string
NODES=$(IFS=, ; echo "${PROCESSED_NODES[*]}")
echo "Nodes of running jobs for user $USERNAME: $NODES"