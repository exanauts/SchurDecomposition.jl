import MPI
MPI.Init()

using Revise
using SchurDecomposition
using LazyArtifacts
using DelimitedFiles
using Argos
using ArgosCUDA
using NLPModels
using MadNLP
using MadNLPGPU

using SparseArrays
using CUDA
using CUDAKernels

# Load GPU utils
CUDA.allowscalar(false)
N_GPU = CUDA.ndevices()

#=
    MPI config
=#
comm = MPI.COMM_WORLD
root = 0

nblk = MPI.Comm_size(comm)
id = MPI.Comm_rank(comm)

CUDA.device!((id+1) % N_GPU)
if id == root
    @info "Init"
    CUDA.memory_status()
    println()
end

is_master = (MPI.Comm_rank(comm) == root)

if is_master
    println("[ARGOS-GPU] Launch optimization on $(nblk) processes.")
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

blk = SchurDecomposition.BlockOPFModel(
    datafile,
    pload, qload,
    id, nscen, nblk;
    device=CUDADevice(),
    comm=comm,
)
if id == root
    @info "Model"
    CUDA.memory_status()
    println()
end

# Instantiate Jacobian with proper values
x0 = NLPModels.get_x0(blk)
nnzj = NLPModels.get_nnzj(blk)
jac = zeros(nnzj)
NLPModels.jac_coord!(blk, x0, jac)


#=
    Step 2: Launch optimization
=#

linear_solver = LapackGPUSolver
verbose = is_master ? MadNLP.DEBUG : MadNLP.ERROR


T = Float64
VI = CuVector{Int}
VT = CuVector{T}
MT = CuMatrix{T}

KKT = SchurDecomposition.SchurKKTSystem{T, VI, VT, MT}
madnlp_options = Dict{Symbol, Any}(
    :tol=>1e-5,
    :max_iter=>max_iter,
    :nlp_scaling=>scaling,
    :print_level=>verbose,
    :linear_solver=>linear_solver,
    :dual_initialized=>true,
    :lapack_algorithm=>MadNLP.CHOLESKY,
)
opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

for i in 1:ntrials
    GC.gc(true)
    CUDA.reclaim()
    ipp = MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
    if id == root
        @info "Optim"
        CUDA.memory_status()
        println()
    end
    MadNLP.solve!(ipp)
end

MPI.Finalize()

