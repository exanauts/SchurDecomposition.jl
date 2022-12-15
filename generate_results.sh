#!/bin/bash

MPI_EXEC=./mpiexecjl
NGPU=1

$MPI_EXEC -n $NGPU julia --project scripts/bench_callbacks.jl
$MPI_EXEC -n $NGPU julia --project scripts/bench_kkt.jl
$MPI_EXEC -n $NGPU julia --project scripts/bench_optim.jl

