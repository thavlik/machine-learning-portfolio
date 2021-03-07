#!/bin/bash
WALLET=0x316f753d58D91640E5Cd3010F821e1CD464e2d3C
while true; do
        ethminer -P stratum1+ssl://${WALLET}@eth-us-east.flexpool.io:5555
        sleep 1s
done
