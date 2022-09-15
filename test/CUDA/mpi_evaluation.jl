using SchurDecomposition
using LazyArtifacts
using DelimitedFiles
using Argos
using ArgosCUDA
using NLPModels
using CUDA
using CUDAKernels

import MPI
MPI.Init()


#=
    MPI config
=#
comm = MPI.COMM_WORLD
root = 0

nblk = MPI.Comm_size(comm)
id = MPI.Comm_rank(comm)
is_master = (MPI.Comm_rank(comm) == root)

if is_master
    println("[ARGOS-GPU] Evaluate callbacks on $(nblk) processes.")
end

#=
    CUDA config
=#
CUDA.device!(id % 2)
CUDA.allowscalar(false)

#=
    Data
=#
include(joinpath("..", "config.jl"))


#=
    Step 1: Load model
=#
pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

# # Create block model
datafile = joinpath(DATA, "$(casename).m")
blk = SchurDecomposition.BlockOPFModel(
    datafile, pload, qload, id, nscen, nblk;
    device=CUDADevice(),
    comm=comm,
)

n = NLPModels.get_nvar(blk)
m = NLPModels.get_ncon(blk)

x0 = NLPModels.get_x0(blk)

#=
    Evaluation of objective
=#
obj = NLPModels.obj(blk, x0)
print("rank = $(MPI.Comm_rank(comm)), x = $(obj)\n")

#=
    Evaluation of constraints
=#
c = zeros(m)
NLPModels.cons!(blk, x0, c)
print("rank = $(MPI.Comm_rank(comm)), c = $(sum(c))\n")

#=
    Evaluation of gradient
=#
g = zeros(n)
NLPModels.grad!(blk, x0, g)
print("rank = $(MPI.Comm_rank(comm)), g = $(sum(g))\n")

#=
    Evaluation of Jacobian
=#
nnzj = NLPModels.get_nnzj(blk)
jac = zeros(nnzj)
NLPModels.jac_coord!(blk, x0, jac)
print("rank = $(MPI.Comm_rank(comm)), J = $(sum(jac))\n")

#=
    Evaluation of Hessian
=#
nnzh = NLPModels.get_nnzh(blk)
hess = zeros(nnzh)
y0 = ones(m)
NLPModels.hess_coord!(blk, x0, y0, hess)
print("rank = $(MPI.Comm_rank(comm)), H = $(sum(hess))\n")

MPI.Finalize()

