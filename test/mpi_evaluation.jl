using SchurDecomposition
using LazyArtifacts
using DelimitedFiles
using Argos
using NLPModels

import MPI
MPI.Init()

comm = MPI.COMM_WORLD
root = 0

const DATA = joinpath(artifact"ExaData", "ExaData")
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

casename = "case9"

nblk = MPI.Comm_size(comm)
id = MPI.Comm_rank(comm)
nscen = 12

shift = 0

pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

# # Create block model
datafile = joinpath(DATA, "$(casename).m")
blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, id, nscen, nblk; comm=comm)

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
jac = zeros(0)
NLPModels.jac_coord!(blk, x0, jac)

#=
    Evaluation of Hessian
=#
hess = zeros(0)
y0 = rand(m)
NLPModels.hess_coord!(blk, x0, y0, hess)

MPI.Finalize()

