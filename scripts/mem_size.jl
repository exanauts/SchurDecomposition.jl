
using DelimitedFiles
using LazyArtifacts
using ExaPF
using Argos
using SchurDecomposition
using NLPModels

include("common.jl")

function memory_size(stack::ExaPF.NetworkStack)
    mem = sizeof(stack.input)
    mem += 8 * sizeof(stack.intermediate.sfp)
    mem += sizeof(stack.ψ)
    mem += sizeof(stack.params)
    return mem
end

function memory_size(jac::ExaPF.AutoDiff.AbstractJacobian)
    mem = memory_size(jac.stack)
    mem += sizeof(jac.coloring)
    mem += sizeof(jac.t1sF)
    return mem
end

function memory_size(hess::ExaPF.AutoDiff.AbstractFullHessian)
    mem = sizeof(hess.map)
    mem += memory_size(hess.stack)
    mem += memory_size(hess.∂stack)
    mem += sizeof(hess.t1sF)
    mem += sizeof(hess.∂t1sF)
    return mem
end

function memory_size(stoch::Argos.StochEvaluator)
    mem = memory_size(stoch.stack)
    mem += memory_size(stoch.∂stack)
    mem += memory_size(stoch.jac)
    mem += memory_size(stoch.hess)

    mem += sizeof(stoch._obj) + sizeof(stoch._cons) + sizeof(stoch._grad_control) + sizeof(stoch._multipliers)

    return mem
end

function memory_size(reduction::Argos.Reduction)
    mem = sizeof(reduction.z) + sizeof(reduction.ψ)
    return mem
end

function memory_size(reduction::Argos.BatchReduction)
    mem = sizeof(reduction.z) + sizeof(reduction.ψ) + sizeof(reduction.tangents)
    return mem
end

function memory_size(kkt::Argos.BieglerKKTSystem)
    mem = sizeof(kkt.aug_com)
    # println("KKT matrix: ", round(mem / 1024^2))
    mem += sizeof(kkt.pr_diag) + sizeof(kkt.du_diag)
    mem += sizeof(kkt.j_V) + sizeof(kkt.h_V)
    mem += sizeof(kkt.mapA) + sizeof(kkt.mapGx) + sizeof(kkt.mapGu)
    mem += sizeof(kkt._wxu1) + sizeof(kkt._wxu2) + sizeof(kkt._wxu3)
    mem += sizeof(kkt._wx1) + sizeof(kkt._wx2)
    mem += sizeof(kkt._wj1)
    mem += sizeof(kkt.con_scale) + sizeof(kkt.jacobian_scaling)
    # println("KKT matrix: ", round(mem / 1024^2))
    mem += memory_size(kkt.reduction)
    # println("KKT matrix: ", round(mem / 1024^2))
    return mem
end

function memory_size(kkt::SchurDecomposition.SchurKKTSystem)
    mem = memory_size(kkt.inner)
    mem += sizeof(kkt.pr_diag)
    mem += sizeof(kkt.du_diag)
    mem += sizeof(kkt._w1)
    mem += sizeof(kkt._w2)
    mem += sizeof(kkt._w3)
    mem += sizeof(kkt._w4)
    mem += sizeof(kkt._w5)
    return mem
end

function build_kkt(blk)
    # Refresh
    x0 = NLPModels.get_x0(blk)
    NLPModels.obj(blk, x0)
    jac = zeros(NLPModels.get_nnzj(blk))
    NLPModels.jac_coord!(blk, x0, jac)

    T = Float64
    VI = Vector{Int}
    VT = Vector{T}
    MT = Matrix{T}
    return SchurDecomposition.SchurKKTSystem{T, VI, VT, MT}(blk)
end

function build_model(casename, nscen)
    nblk = 1
    id = 0

    datafile = joinpath(DATA, "$(casename).m")
    model = PolarForm(datafile, DEVICE)
    pload, qload = generate_loads(model, nscen, 0.0)

    # Create block model
    return SchurDecomposition.BlockOPFModel(model, pload, qload, 0, nscen, 2)
end

function how_much_memory(casename, nscen)
    blk = build_model(casename, nscen)
    kkt = build_kkt(blk)

    model = Argos.backend(blk.model)
    println("MODEL : ", round(memory_size(model) / 1024^2), "mb")
    println("KKT :   ", round(memory_size(kkt) / 1024^2), "mb")
end

function how_much_memory_70k(casename, nscen)
    nblk = 1
    id = 0

    casename = "case_ACTIVSg25k"
    pload = readdlm(joinpath(DEMANDS, "$(casename)_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_60.Qd")) ./ 100

    # Create block model
    datafile = joinpath(DATA, "matpower", "$(casename).m")
    blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, id, nscen, nblk)
    kkt = build_kkt(blk)

    model = Argos.backend(blk.model)
    println("MODEL : ", round(memory_size(model) / 1024^2), "mb")
    println("KKT :   ", round(memory_size(kkt) / 1024^2), "mb")
end

function bench_memory(casename; nscens=[10, 20, 30, 60, 120, 240])
    nblk, id = 1, 0

    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)

    nexp = length(nscens)
    results = zeros(nexp, 2)
    for (i, k) in enumerate(nscens)
        pload, qload = generate_loads(model, k, 0.0)
        # Create block model
        blk = SchurDecomposition.BlockOPFModel(model, pload, qload, 0, k, 2)
        kkt = build_kkt(blk)

        sto = Argos.backend(blk.model)
        results[i, 1] = memory_size(sto) / 1024^2
        results[i, 2] = memory_size(kkt) / 1024^2
    end
    return results
end

