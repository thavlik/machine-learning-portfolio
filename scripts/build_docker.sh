#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
IMAGE=thavlik/machine-learning-portfolio:latest
NAME=ml
docker build -t $IMAGE .
docker push $IMAGE