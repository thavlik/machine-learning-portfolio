#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
logdir() {
    if [ -f "/etc/wsl.conf" ]; then
        echo "$(wslpath -a logs)"
    else
        echo "$(pwd)/logs"
    fi
}
name=hometb
docker kill $name || true 2>/dev/null
docker container stop $name || true 2>/dev/null
docker container rm $name || true 2>/dev/null
docker run \
    -d \
    --rm \
    --name $name \
    -p 6006:6006 \
    -v $(logdir):/logs \
    tensorflow/tensorflow:latest-jupyter \
    tensorboard \
        --logdir /logs \
        --host 0.0.0.0 \
        --port 6006
