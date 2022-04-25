
using Revise
using Test
using LazyArtifacts
using DelimitedFiles
using Argos
using NLPModels
using FiniteDiff
using SchurDecomposition

const DATA = joinpath(artifact"ExaData", "ExaData")
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

function test_block_opf_model(casename)
    nblk = 2
    nscen = 10

    shift = 0

    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

    # Create block model
    datafile = joinpath(DATA, "$(casename).m")
    blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, 1, nscen, nblk)

    @test isa(blk, NLPModels.AbstractNLPModel)
    return blk
end

@testset "BlockOPFModel" begin
    blk = test_block_opf_model("case9")
end

