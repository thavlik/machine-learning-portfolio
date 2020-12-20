#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
helm upgrade traefik traefik/traefik -n traefik -f traefik.yaml
kubectl get pod -n traefik -w
