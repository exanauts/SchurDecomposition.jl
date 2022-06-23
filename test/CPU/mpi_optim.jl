using SchurDecomposition
using LazyArtifacts
using DelimitedFiles
using Argos
using NLPModels
using MadNLP

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
    println("[ARGOS-CPU] Launch optimization on $(nblk) processes.")
end

#=
    Data
=#
include(joinpath("..", "config.jl"))


#=
    Step 1: Load model
=#

datafile = joinpath(DATA, "$(casename).m")
pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, id, nscen, nblk; comm=comm)

# Instantiate Jacobian with proper values
x0 = NLPModels.get_x0(blk)
nnzj = NLPModels.get_nnzj(blk)
jac = zeros(nnzj)
NLPModels.jac_coord!(blk, x0, jac)

#=
    Step 2: Launch optimization
=#

linear_solver=MadNLPLapackCPU
verbose = is_master ? MadNLP.DEBUG : MadNLP.ERROR

options = Dict{Symbol, Any}(
    :tol=>1e-5,
    :max_iter=>max_iter,
    :nlp_scaling=>scaling,
    :print_level=>verbose,
    :linear_solver=>linear_solver,
    :dual_initialized=>true,
)
madopt = MadNLP.Options(linear_solver=linear_solver)
MadNLP.set_options!(madopt, options, Dict())

KKT = SchurDecomposition.SchurKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
ipp = MadNLP.InteriorPointSolver{KKT}(blk, madopt; option_linear_solver=options)
MadNLP.optimize!(ipp)

MPI.Finalize()
