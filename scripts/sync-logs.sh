#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"/..
scp -r -P 34759 root@ssh5.vast.ai:/machine-learning-portfolio/logs .