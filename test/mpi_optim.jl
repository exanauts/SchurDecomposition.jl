using SchurDecomposition
using LazyArtifacts
using DelimitedFiles
using Argos
using NLPModels
using MadNLP

import MPI

#=
    MPI config
=#
MPI.Init()
comm = MPI.COMM_WORLD
root = 0

nblk = MPI.Comm_size(comm)
id = MPI.Comm_rank(comm)

is_master = (MPI.Comm_rank(comm) == root)

if is_master
    println("[ARGOS] Launch optimization on $(nblk) processes.")
end

#=
    Data
=#
const DATA = "../matpower/data/" #joinpath(artifact"ExaData", "ExaData")
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

casename = "case9"
nscen = 12

#=
    Step 1: Load model
=#

datafile = joinpath(DATA, "$(casename).m")
pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, id, nscen, nblk; comm=comm)

# Instantiate Jacobian with proper values
x0 = NLPModels.get_x0(blk)
println(sum(x0))
nnzj = NLPModels.get_nnzj(blk)
jac = zeros(nnzj)
NLPModels.jac_coord!(blk, x0, jac)

#=
    Step 2: Launch optimization
=#
scaling = false
max_iter = 20
linear_solver=MadNLPLapackCPU
if is_master
    verbose = MadNLP.DEBUG
else
    verbose = MadNLP.ERROR
end

options = Dict{Symbol, Any}(
    :tol=>1e-5,
    :max_iter=>max_iter,
    :nlp_scaling=>scaling,
    :print_level=>verbose,
    :linear_solver=>linear_solver,
    :dual_initialized=>true,
)
madopt = MadNLP.Options(linear_solver=linear_solver)
MadNLP.set_options!(madopt, options,Dict())

KKT = SchurDecomposition.SchurKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
ipp = MadNLP.InteriorPointSolver{KKT}(blk, madopt; option_linear_solver=options)
MadNLP.optimize!(ipp)

MPI.Finalize()
