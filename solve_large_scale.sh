#!/bin/bash

MPI_EXEC=./mpiexecjl
NGPU=8 # number of GPUs should be at least 8

$MPI_EXEC -n $NGPU julia --project scripts/bench_optim_large_scale.jl

