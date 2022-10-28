using Argos
using CUDA
using DelimitedFiles
using ExaPF
using KernelAbstractions
using LazyArtifacts
using MadNLP
using MPI
using NLPModels
using Random
using SchurDecomposition

#=
    PARAMETERS
=#
const DATA = joinpath(artifact"ExaData", "ExaData")
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")
RESULTS_DIR = "results"
SAVE_RESULTS = true

#=
    DEVICE
=#
DEVICE = CPU()
LAPACK_SOLVER = LapackCPUSolver

#=
    CUDA config
=#
if CUDA.has_cuda()
    CUDA.device!(id % 2)
    CUDA.allowscalar(false)
end

#=
    MPI config
=#
MPI.Init()
comm = MPI.COMM_WORLD
root = 0

nblk = MPI.Comm_size(comm)
id = MPI.Comm_rank(comm)
is_master = (MPI.Comm_rank(comm) == root)
#

#=
    UTILS FUNCTION
=#
function refresh_memory()
    GC.gc(true)
    CUDA.has_cuda() && CUDA.reclaim()
    return
end

function import_loads(casename)
    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100
    return pload, qload
end

function generate_loads(model, nscen, magnitude)
    nbus = get(model, ExaPF.PS.NumberOfBuses())
    stack = ExaPF.NetworkStack(model)
    pload_det = stack.pload
    qload_det = stack.qload

    has_load = (pload_det .> 0)

    Random.seed!(1)
    pload = magnitude .* (randn(nbus, nscen) .* has_load) .+ pload_det
    qload = magnitude .* (randn(nbus, nscen) .* has_load) .+ qload_det
    return pload, qload
end

function load_model(model::ExaPF.PolarForm, nscen, comm)
    pload, qload = generate_loads(model, nscen, 0.0)
    return SchurDecomposition.BlockOPFModel(
        model, pload, qload, id, nscen, nblk;
        comm=comm,
    )
end

function instantiate_model!(blk::SchurDecomposition.BlockOPFModel)
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    return
end

function build_solver(blk::SchurDecomposition.BlockOPFModel)
    madnlp_options = Dict{Symbol, Any}(
        :dual_initialized=>true,
        :lapack_algorithm=>MadNLP.CHOLESKY,
        :linear_solver=>LAPACK_SOLVER,
        :max_iter=>250,
        :print_level=>MadNLP.ERROR,
        :tol=>1e-5,
    )
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)

    KKT = SchurDecomposition.SchurKKTSystem{Float64, Vector{Int}, Vector{Float64}, Matrix{Float64}}
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

