
using Revise
using Test
using LazyArtifacts
using DelimitedFiles
using Argos
using NLPModels
using FiniteDiff
using SchurDecomposition

const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

function test_block_opf_model(casename)
    nblk = 2
    nscen = 5

    shift = 0
    stochs = []
    for idx in 1:nblk
        span = shift+1:shift+nscen

        pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd"))[:, span] ./ 100
        qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd"))[:, span] ./ 100
        push!(
            stochs,
            Argos.StochEvaluator("../matpower/data/$(casename).m", pload, qload)
        )
        shift += nscen
    end

    # Create block model
    blk = SchurDecomposition.BlockOPFModel(stochs)

    @test isa(blk, NLPModels.AbstractNLPModel)

    n = NLPModels.get_nvar(blk)
    m = NLPModels.get_ncon(blk)
    x0 = NLPModels.get_x0(blk)
    c = zeros(m)
    g = zeros(n)

    @test length(x0) == n
    @test isa(NLPModels.obj(blk, x0), Float64)

    # Test cons
    NLPModels.cons!(blk, x0, c)

    # Test gradient
    eval_f(x) = NLPModels.obj(blk, x)
    NLPModels.grad!(blk, x0, g)
    @test g ≈ FiniteDiff.finite_difference_gradient(eval_f, x0)
    return blk
end

@testset "BlockOPFModel" begin
    blk = test_block_opf_model("case9")
end

