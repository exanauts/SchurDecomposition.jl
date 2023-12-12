
using Revise
using Test
using LazyArtifacts
using LinearAlgebra
using DelimitedFiles
using Argos
using MadNLP
using MadNLPHSL
using NLPModels
using FiniteDiff
using SchurDecomposition
using SparseArrays

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

function initialize_kkt!(kkt, cb)
    MadNLP.initialize!(kkt)
    # Compute initial values for Hessian and Jacobian
    x0 = NLPModels.get_x0(cb.nlp)
    y0 = NLPModels.get_y0(cb.nlp)
    # Update Jacobian manually
    jac = MadNLP.get_jacobian(kkt)
    MadNLP._eval_jac_wrapper!(cb, x0, jac)
    MadNLP.compress_jacobian!(kkt)
    # Update Hessian manually
    hess = MadNLP.get_hessian(kkt)
    MadNLP._eval_lag_hess_wrapper!(cb, x0, y0, hess)
    MadNLP.compress_hessian!(kkt)

    MadNLP._set_aug_diagonal!(kkt)
    MadNLP.build_kkt!(kkt)
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

    # Create KKT
    linear_solver = LapackCPUSolver

    T = Float64
    VI = Vector{Int}
    VT = Vector{T}
    MT = Matrix{T}
    KKT = SchurDecomposition.SchurKKTSystem{T, VI, VT, MT}

    solver = MadNLP.MadNLPSolver(
        blk;
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        kkt_system=KKT,
    )

    @test isa(solver.kkt, MadNLP.AbstractReducedKKTSystem)

    # Test MadNLP wrapper
    MadNLP.build_kkt!(solver.kkt)
    MadNLP.factorize_wrapper!(solver)
    # MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)

    return
end

function test_parallel_kkt(casename)
    # We consider only one block: in that case ParallelKKTSystem
    # should be equivalent to the classical SparseKKTSystem.
    nblk = 1
    nscen = 10
    linear_solver = LapackCPUSolver
    options = MadNLP.MadNLPOptions(; linear_solver=linear_solver)
    options_linear_solver = MadNLP.default_options(linear_solver)

    pload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Pd")) ./ 100
    qload = readdlm(joinpath(DEMANDS, "$(casename)_onehour_60.Qd")) ./ 100
    datafile = joinpath(DATA, "$(casename).m")
    blk = SchurDecomposition.BlockOPFModel(datafile, pload, qload, 0, nscen, nblk)

    # Problem's dimensions
    nx = blk.model.nlp.nx
    nu = blk.model.nlp.nu
    n, m = NLPModels.get_nvar(blk.model), NLPModels.get_ncon(blk.model)
    ind_cons = MadNLP.get_index_constraints(
        blk.model,
        options.fixed_variable_treatment,
        options.equality_treatment,
    )
    n_ineq = length(ind_cons.ind_ineq)
    nki = n - nu + n_ineq + m
    nw = n + n_ineq + m
    ind_ki = vcat(1:nx, nx+nu+1:n+n_ineq+m)

    cnt = MadNLP.MadNLPCounters(; start_time=time())

    @testset "Local KKT" begin
        inner_cb = MadNLP.create_callback(
            MadNLP.SparseCallback,
            blk.model,
            options,
        )
        # Build local KKT system
        kkt = MadNLP.create_kkt_system(
            MadNLP.SparseKKTSystem, inner_cb, options, options_linear_solver, cnt, ind_cons,
        )
        initialize_kkt!(kkt, inner_cb)
        MadNLP.factorize!(kkt.linear_solver)
        W = kkt.aug_com


        @test size(W, 1) == n + n_ineq + m

        # Test splitting of local KKT system
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

        Ki_ref = W[ind_ki, ind_ki]
        Ki.nzval .= W.nzval[res.mapKi]
        @test Ki == Ki_ref

        Wxu = W[nx+1:nx+nu, 1:nx]
        Gu = W[nx+nu+1:nw, nx+1:nx+nu]
        Bi_ref = [Wxu'; Gu]
        Bi.nzval .= W.nzval[mapBi_new]
        @test Bi == Bi_ref

        b = MadNLP.UnreducedKKTVector(kkt)
        x = MadNLP.UnreducedKKTVector(kkt)
        MadNLP.full(x) .= 1.0
        MadNLP.solve!(kkt, x)
        mul!(b, kkt, x)
        @test MadNLP.full(b) ≈ ones(length(b)) atol=1e-6
    end

    @testset "Global KKT" begin
        ind_cons = MadNLP.get_index_constraints(
            blk,
            options.fixed_variable_treatment,
            options.equality_treatment,
        )
        cb = MadNLP.create_callback(
            MadNLP.SparseCallback,
            blk,
            options,
        )

        T = Float64
        KKT = SchurDecomposition.ParallelKKTSystem{T, Vector{T}, Matrix{T}}
        pkkt = MadNLP.create_kkt_system(
            KKT, cb, options, options_linear_solver, cnt, ind_cons,
        )

        initialize_kkt!(pkkt, cb)
        MadNLP.factorize!(pkkt.linear_solver)

        # Test definition of matrices
        Wref = pkkt.local_kkt.aug_com
        K0_ref = Wref[nx+1:nx+nu, nx+1:nx+nu]
        Ki_ref = Wref[ind_ki, ind_ki]
        Wxu = Wref[nx+1:nx+nu, 1:nx]
        Gu = Wref[nx+nu+1:nw, nx+1:nx+nu]
        Bi_ref = [Wxu'; Gu]

        @test pkkt.K0 == K0_ref
        @test pkkt.Ki == Ki_ref
        @test pkkt.Bi == Bi_ref

        # Test backsolve
        b = MadNLP.UnreducedKKTVector(pkkt)
        x = MadNLP.UnreducedKKTVector(pkkt)
        rhs = ones(length(x))
        MadNLP.full(x) .= rhs

        @test size(pkkt, 1) == length(MadNLP.primal_dual(b))

        # TODO: segfault
        MadNLP.solve!(pkkt, x)
        mul!(b, pkkt, x)
        @test MadNLP.full(b) ≈ rhs atol=1e-6
    end
    return
end

@testset "BlockOPFModel" begin
    test_block_opf_model("case9")
    # test_schur_kkt("case9")
    test_parallel_kkt("case9")
end

