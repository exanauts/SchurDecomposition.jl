using ExaPF
using Random
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

NBENCH = 10

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

function import_loads(casename)
    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100
    return pload, qload
end

function generate_loads(datafile, nscen, magnitude)
    polar = ExaPF.PolarForm(datafile)
    nbus = get(polar, ExaPF.PS.NumberOfBuses())
    stack = ExaPF.NetworkStack(polar)
    pload_det = stack.pload
    qload_det = stack.qload

    has_load = (pload_det .> 0)

    Random.seed!(1)
    pload = magnitude .* (randn(nbus, nscen) .* has_load) .+ pload_det
    qload = magnitude .* (randn(nbus, nscen) .* has_load) .+ qload_det
    return pload, qload
end

function benchmark_callbacks(blk)
    n = NLPModels.get_nvar(blk)
    m = NLPModels.get_ncon(blk)
    x0 = NLPModels.get_x0(blk)

    obj = NLPModels.obj(blk, x0)
    tobj = @elapsed for i in 1:NBENCH
        NLPModels.obj(blk, x0)
    end
    is_master && println("  TOBJ = ", tobj)

    c = zeros(m)
    NLPModels.cons!(blk, x0, c)
    tcons = @elapsed for i in 1:NBENCH
        NLPModels.cons!(blk, x0, c)
    end
    is_master && println("  TCONS = ", tcons)

    g = zeros(n)
    NLPModels.grad!(blk, x0, g)
    tgrad = @elapsed for i in 1:NBENCH
        NLPModels.grad!(blk, x0, g)
    end
    is_master && println("  TGRAD = ", tgrad)

    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    tjac = @elapsed for i in 1:NBENCH
        NLPModels.jac_coord!(blk, x0, jac)
    end
    is_master && println("  TJAC = ", tjac)

    nnzh = NLPModels.get_nnzh(blk)
    hess = zeros(nnzh)
    y0 = ones(m)
    NLPModels.hess_coord!(blk, x0, y0, hess)
    thess = @elapsed for i in 1:NBENCH
        NLPModels.hess_coord!(blk, x0, y0, hess)
    end
    is_master && println("  THESS = ", thess)
end

function load_model(datafile, nscen)
    pload, qload = generate_loads(datafile, nscen, 0.0)
    return SchurDecomposition.BlockOPFModel(
        datafile, pload, qload, id, nscen, nblk;
        device=CUDADevice(),
        comm=comm,
    )
end

cases = [
    "case1354pegase",
    # "case_ACTIVSg2000",
    # "case_ACTIVSg10k",
    # "case_ACTIVSg200",
]

for case in cases, nscen in [10, 20, 30, 60, 120, 240]
    is_master && println()
    is_master && println("-"^50)
    is_master && println("$(case) nscen: ", nscen)
    datafile = joinpath(DATA, "$(case).m")
    blk = load_model(datafile, nscen)
    benchmark_callbacks(blk)
    GC.gc(true)
    CUDA.reclaim()
end

MPI.Finalize()
