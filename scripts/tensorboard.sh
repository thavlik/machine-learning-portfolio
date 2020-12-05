#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
dir=$(pwd)
docker run \
    -it \
    -p 6006:6006 \
    -v $(wslpath -w logs):/logs  \
    tensorflow/tensorflow:latest-jupyter \
    tensorboard \
        --logdir /logs \
        --host 0.0.0.0 \
        --port 6006