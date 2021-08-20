#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
dir=$(pwd)
docker container stop hometb || true
docker container rm hometb || true
logdir=$(wslpath -a logs)
echo "logdir=$logdir"
docker run \
    -d \
    --name hometb \
    -p 6007:6006 \
    -v $logdir:/logs \
    tensorflow/tensorflow:latest-jupyter \
    tensorboard \
        --logdir /logs \
        --host 0.0.0.0 \
        --port 6006
