#!/bin/bash

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT
# PJ-lab eth interface == bond;
# Ali-dsw eth interface == net0;
export NCCL_SOCKET_IFNAME=net0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1
export NCCL_IB_HCA='mlx5_2'

export CUDA_VISIBLE_DEVICES='2,3'

mpirun -n 2 --allow-run-as-root bwtest.out