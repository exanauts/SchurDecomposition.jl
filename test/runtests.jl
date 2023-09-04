
using Revise
using Test
using LazyArtifacts
using DelimitedFiles
using Argos
using MadNLP
using NLPModels
using FiniteDiff
using SchurDecomposition

const DATA = joinpath(artifact"ExaData", "ExaData")
const DEMANDS = joinpath(artifact"ExaData", "ExaData", "mp_demand")

function test_block_opf_model(casename)
    nblk = 2
    nscen = 10

    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

    # Create block model
    datafile = joinpath(DATA, "$(casename).m")
    blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, 1, nscen, nblk)

    @test isa(blk, NLPModels.AbstractNLPModel)

    n = NLPModels.get_nvar(blk)
    m = NLPModels.get_ncon(blk)
    x0 = NLPModels.get_x0(blk)

    NLPModels.obj(blk, x0)

    g = zeros(n)
    NLPModels.grad!(blk, x0, g)
    # Check if gradient is correct
    g_fd = FiniteDiff.finite_difference_gradient(x -> NLPModels.obj(blk, x), x0)
    @test g_fd ≈ g

    c = zeros(m)
    NLPModels.cons!(blk, x0, c)
    return
end

function test_schur_kkt(casename)
    nblk = 2
    nscen = 10
    id = 0

    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100

    # Create block model
    datafile = joinpath(DATA, "$(casename).m")
    blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, id, nscen, nblk)

    x0 = NLPModels.get_x0(blk)
    NLPModels.obj(blk, x0)
    jac = zeros(NLPModels.get_nnzj(blk))
    NLPModels.jac_coord!(blk, x0, jac)

    T = Float64
    VI = Vector{Int}
    VT = Vector{T}
    MT = Matrix{T}
    kkt = SchurDecomposition.SchurKKTSystem{T, VI, VT, MT}(blk)

    @test isa(kkt, MadNLP.AbstractReducedKKTSystem)
    return
end

function test_parallel_kkt(casename)
    nblk = 2
    nscen = 10

    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100
    datafile = joinpath(DATA, "$(casename).m")
    blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, 1, nscen, nblk)

    # Problem's dimensions
    nx = blk.model.nlp.nx
    nu = blk.model.nlp.nu
    n, m = NLPModels.get_nvar(blk.model), NLPModels.get_ncon(blk.model)
    ind_cons = MadNLP.get_index_constraints(blk.model)
    n_ineq = length(ind_cons.ind_ineq)

    # Build local KKT system
    T = Float64
    VT = Vector{Float64}
    SMT = SparseMatrixCSC{T, Int32}
    QN = MadNLP.ExactHessian{T, VT}
    kkt = MadNLP.SparseKKTSystem{T, VT, SMT, QN}(blk.model, ind_cons)
    MadNLP.build_kkt!(kkt)
    W = kkt.aug_com
    nw = size(W, 1)

    @test nw == n + n_ineq + m

    # Test splitting of local KKT system
    nki = n - nu + n_ineq + m
    res = SchurDecomposition.split_kkt_system(W, nx, nu)

    K0 = sparse(res.i_K0, res.j_K0, convert.(Float64, res.mapK0))
    Ki = sparse(res.i_Ki, res.j_Ki, convert.(Float64, res.mapKi))
    Bi = sparse(res.i_Bi, res.j_Bi, convert.(Float64, res.mapBi), nki, nu)

    mapK0_new = K0.nzval .|> Int
    mapKi_new = Ki.nzval .|> Int
    mapBi_new = Bi.nzval .|> Int

    # Check we have scanned all the elements in W
    @assert max(res.mapBi[end], res.mapKi[end], res.mapK0[end]) == nnz(W)

    @test size(K0) == (nu, nu)
    @test size(Ki) == (nki, nki)
    @test size(Bi) == (nki, nu)

    W.nzval .= rand(nnz(W))
    K0_ref = W[nx+1:nx+nu, nx+1:nx+nu]

    K0.nzval .= W.nzval[res.mapK0]
    @test K0 == K0_ref

    ind_ki = vcat(1:nx, nx+nu+1:n+n_ineq+m)
    Ki_ref = W[ind_ki, ind_ki]
    Ki.nzval .= W.nzval[res.mapKi]
    @test Ki == Ki_ref

    Wxu = W[nx+1:nx+nu, 1:nx]
    Gu = W[nx+nu+1:nw, nx+1:nx+nu]
    Bi_ref = [Wxu'; Gu]
    Bi.nzval .= W.nzval[mapBi_new]
    @test Bi == Bi_ref

    # Test ParallelKKTSystem
    KKT = SchurDecomposition.ParallelKKTSystem{T, VT, Matrix{Float64}}
    pkkt = KKT(blk, ind_cons)
    MadNLP.initialize!(pkkt)
    MadNLP.build_kkt!(pkkt)

    Wref = pkkt.inner.aug_com
    K0_ref = Wref[nx+1:nx+nu, nx+1:nx+nu]
    Ki_ref = Wref[ind_ki, ind_ki]
    Wxu = Wref[nx+1:nx+nu, 1:nx]
    Gu = Wref[nx+nu+1:nw, nx+1:nx+nu]
    Bi_ref = [Wxu'; Gu]

    @test pkkt.K0 == K0_ref
    @test pkkt.Ki == Ki_ref
    @test pkkt.Bi == Bi_ref
    return
end

@testset "BlockOPFModel" begin
    test_block_opf_model("case9")
    test_schur_kkt("case9")
    test_parallel_kkt("case9")
end

