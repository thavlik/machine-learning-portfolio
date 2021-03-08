#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
dir=$(pwd)
docker container stop hometb2 || true
docker container rm hometb2 || true
docker run \
    -d \
    --name hometb2 \
    -p 6008:6006 \
    -v $(wslpath -w logs_remote)/:/logs \
    tensorflow/tensorflow:latest-jupyter \
    tensorboard \
        --logdir /logs \
        --host 0.0.0.0 \
        --port 6006
