#!/bin/bash

export MPICH_GPU_SUPPORT_ENABLED=0
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_MPI_BINARY=""
unset JULIA_MPI_PATH
unset MPI_GPU_SUPPORT_ENABLED
export JULIA_MPI_HAS_CUDA=0
# Uncomment for Polaris MPI with CUDA
export JULIA_MPI_BINARY="system"
export MPICH_GPU_SUPPORT_ENABLED=1
export JULIA_MPI_PATH=/opt/cray/pe/mpich/8.1.16/ofi/nvidia/20.7
export JULIA_MPI_HAS_CUDA=1

export JULIA_CUDA_USE_BINARYBUILDER=false
julia --project -e 'using Pkg; Pkg.build("MPI"; verbose=true)'
julia --project -e 'using Pkg; Pkg.build("CUDA"; verbose=true)'
julia --project -e 'using MPI ; MPI.install_mpiexecjl(force=true ;destdir=".")'
export OMP_NUM_THREADS=1
