#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
scp -r -P 34968 root@ssh5.vast.ai:/machine-learning-portfolio/logs logs_remote