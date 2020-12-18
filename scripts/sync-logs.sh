#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
scp -r -P 20541 root@ssh4.vast.ai:/machine-learning-portfolio/logs .
