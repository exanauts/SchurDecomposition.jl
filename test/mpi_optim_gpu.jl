import MPI
MPI.Init()

using Revise
using SchurDecomposition
using LazyArtifacts
using DelimitedFiles
using Argos
using NLPModels
using MadNLP
using MadNLPGPU

using CUDA
using CUDAKernels

# Load GPU utils
include("/home/fpacaud/exa/Argos/test/cusolver.jl")
# We are living at the edge
CUDA.allowscalar(false)
N_GPU = 2

#=
    MPI config
=#
comm = MPI.COMM_WORLD
root = 0

nblk = MPI.Comm_size(comm)
id = MPI.Comm_rank(comm)

CUDA.device!((id+1) % N_GPU)

is_master = (MPI.Comm_rank(comm) == root)

if is_master
    println("[ARGOS-GPU] Launch optimization on $(nblk) processes.")
end

#=
    Data
=#
include("config.jl")


#=
    Step 1: Load model
=#

datafile = joinpath(DATA, "$(casename).m")
pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

blk = SchurDecomposition.BlockOPFModel(
    datafile,
    pload, qload,
    id, nscen, nblk;
    device=CUDADevice(),
    comm=comm,
)

# Instantiate Jacobian with proper values
x0 = NLPModels.get_x0(blk)
nnzj = NLPModels.get_nnzj(blk)
jac = zeros(nnzj)
NLPModels.jac_coord!(blk, x0, jac)

#=
    Step 2: Launch optimization
=#

linear_solver = MadNLPLapackGPU
verbose = is_master ? MadNLP.DEBUG : MadNLP.ERROR

options = Dict{Symbol, Any}(
    :tol=>1e-5,
    :max_iter=>max_iter,
    :nlp_scaling=>scaling,
    :print_level=>verbose,
    :linear_solver=>linear_solver,
    :lapackgpu_algorithm=>MadNLPLapackGPU.CHOLESKY,
    :dual_initialized=>true,
)

madopt = MadNLP.Options(linear_solver=linear_solver)
MadNLP.set_options!(madopt, options, Dict())

T = Float64
VI = CuVector{Int}
VT = CuVector{T}
MT = CuMatrix{T}

KKT = SchurDecomposition.SchurKKTSystem{T, VI, VT, MT}
ipp = MadNLP.InteriorPointSolver{KKT}(blk, madopt; option_linear_solver=options)
MadNLP.optimize!(ipp)

MPI.Finalize()
