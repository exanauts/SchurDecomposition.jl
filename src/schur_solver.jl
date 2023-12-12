
abstract type AbstractSchurSolver end

#=
    Ma57
=#

struct Ma57SchurSolver{T} <: AbstractSchurSolver
    solver::MadNLPHSL.Ma57Solver
    V::Matrix{T}
    lwork::Vector{T}
end

function Ma57SchurSolver(Ki, Bi, K0; max_rhs=16)
    nki = size(Ki, 1)
    n_coupling = size(K0, 1)
    # Check consistency
    @assert size(K0, 2) == size(Bi, 2) == n_coupling
    @assert size(Ki, 2) == size(Bi, 1) == nki

    solver = MadNLPHSL.Ma57Solver(Ki)

    nrhs = min(max_rhs, n_coupling)

    # Build buffers
    V = zeros(nki, nrhs)
    lwork = zeros(nki * nrhs)
    return Ma57SchurSolver(solver, V, lwork)
end

function solve!(rsol::Ma57SchurSolver, X::AbstractMatrix)
    M = rsol.solver
    n, m = size(X)
    @assert m == size(rsol.V, 2)
    nrhs = m

    rsol.V .= X
    # Solve linear system with multiple right-hand-sides
    MadNLPHSL.HSL.ma57cd(
        one(Int32),
        Int32(M.csc.n),
        M.fact,
        M.lfact,
        M.ifact,
        M.lifact,
        Int32(nrhs),
        rsol.V,
        Int32(n),
        rsol.lwork,
        Int32(n * nrhs),
        M.iwork,
        M.icntl,
        M.info,
    )
    M.info[1] < 0 && throw(MadNLP.SolveException())
    return rsol.V
end

function solve!(rsol::Ma57SchurSolver, x::AbstractVector)
    return MadNLP.solve!(rsol.solver, x)
end

function schur_solve!(rsol::Ma57SchurSolver, S, Ki, Bi, K0)
    # Update factorization
    copyto!(rsol.solver.csc.nzval, Ki.nzval)
    MadNLP.factorize!(rsol.solver)

    nrhs = size(rsol.V, 2)
    nu = size(K0, 1)
    N = div(nu, nrhs, RoundDown)
    for i in 1:N
        X = view(Bi, :, (i-1)*nrhs+1:i*nrhs)
        Sv = view(S, :, (i-1)*nrhs+1:i*nrhs)
        KBi = solve!(rsol, X)            #  KBi = Kᵢ⁻¹ Bᵢ
        mul!(Sv, Bi', KBi, -1.0, 0.0)    #  S += - Bᵢᵀ Kᵢ⁻¹ Bᵢ
    end
    last_batch = nu - N * nrhs
    if last_batch > 0
        span = (nu-nrhs+1):nu
        X = view(Bi, :, span)
        Sv = view(S, :, span)
        KBi = solve!(rsol, X)            #  KBi = Kᵢ⁻¹ Bᵢ
        mul!(Sv, Bi', KBi, -1.0, 0.0)    #  S += - Bᵢᵀ Kᵢ⁻¹ Bᵢ
    end

    S .+= Symmetric(K0, :L)              #  S = K₀
    return
end

#=
    Pardiso
=#

mutable struct PardisoSolver{T}
    n::Int
    csc::SparseMatrixCSC{T,Int32}
    w::Vector{T}
    pt::Vector{Int}
    iparm::Vector{Int32}
    dparm::Vector{T}
    perm::Vector{Int32}
    mtype::Ref{Int32}
    mnum::Ref{Int32}
    maxfct::Ref{Int32}
    msglvl::Ref{Int32}
    err::Ref{Int32}
    cnt::Ref{Int32}
end

function PardisoSolver(csc::SparseMatrixCSC{T,Int32}; check_matrix=true) where {T}
    n = size(csc, 1)
    w = zeros(T, n)
    pt = zeros(Int, 64)
    iparm = zeros(Int32, 64)
    dparm = zeros(T, 64)
    perm = zeros(Int32, n)

    # Set iparams
    iparm[1] = 1
    iparm[2] = 2  # Metis ordering
    iparm[3] = 1  # num procs
    iparm[8] = 8  # number of IR steps
    iparm[10] = 12 # pivot perturbation
    iparm[11] = 0 # disable scaling
    iparm[13] = 2 # matching strategy
    iparm[21] = 3 # bunch-kaufman pivotin
    iparm[24] = 0 # parallel factorization
    iparm[25] = 0 # parallel solv
    iparm[29] = 0 # parallel solv
    iparm[33] = 1 # compute determinant

    # Control params
    msglvl = Ref{Int32}(0)
    mtype = Ref{Int32}(-2)
    err = Ref{Int32}(0)
    maxfct = Ref{Int32}(1)
    mnum = Ref{Int32}(1)

    # Direct solver
    solver = Ref{Int32}(0)

    # Check that matrix parses correctly in Pardiso
    if check_matrix
        LibPardiso.pardiso_chkmatrix(
            mtype,
            Ref(Int32(n)),
            csc.nzval,
            csc.colptr,
            csc.rowval,
            err,
        )
        @assert err[] == 0
    end

    # Init Pardiso
    LibPardiso.pardisoinit_d(pt, mtype, solver, iparm, dparm, err)
    @assert err[] == 0

    phase = Ref(Int32(12))
    nrhs = Ref(Int32(1))
    LibPardiso.pardiso_d(
        pt,
        maxfct,
        mnum,
        mtype,
        phase,
        Ref{Int32}(n),
        csc.nzval,
        csc.colptr,
        csc.rowval,
        perm,
        nrhs,
        iparm,
        msglvl,
        T[],
        T[],
        err,
        dparm,
    )
    @assert err[] == 0
    M = PardisoSolver{T}(
        n,
        csc,
        w,
        pt,
        iparm,
        dparm,
        perm,
        mtype,
        mnum,
        maxfct,
        msglvl,
        err,
        Ref(Int32(0)),
    )
    finalizer(finalize!, M)
    return M
end

function factorize!(M::PardisoSolver{T}) where {T}
    if M.cnt[] == 0 # Recompute symbolic factorization at first iteration.
        phase = Ref(Int32(12))
    else
        phase = Ref(Int32(22))
    end
    nrhs = Ref(Int32(1))
    LibPardiso.pardiso_d(
        M.pt,
        M.maxfct,
        M.mnum,
        M.mtype,
        phase,
        Ref{Int32}(M.n),
        M.csc.nzval,
        M.csc.colptr,
        M.csc.rowval,
        M.perm,
        nrhs,
        M.iparm,
        M.msglvl,
        T[],
        T[],
        M.err,
        M.dparm,
    )
    @assert M.err[] == 0
    M.cnt[] += 1
    return M
end

function solve!(M::PardisoSolver{T}, rhs::Vector{T}) where {T}
    @assert length(rhs) == M.n
    phase = Ref(Int32(33))
    nrhs = Ref(Int32(1))
    M.iparm[6] = 1 # Unique RHS
    LibPardiso.pardiso_d(
        M.pt,
        M.maxfct,
        M.mnum,
        M.mtype,
        phase,
        Ref{Int32}(M.n),
        M.csc.nzval,
        M.csc.colptr,
        M.csc.rowval,
        M.perm,
        nrhs,
        M.iparm,
        M.msglvl,
        rhs,
        M.w,
        M.err,
        M.dparm,
    )
    @assert M.err[] == 0
end

function finalize!(M::PardisoSolver{T}) where {T}
    phase = Ref(Cint(-1))
    LibPardiso.pardiso_d(
        M.pt,
        M.maxfct,
        M.mnum,
        M.mtype,
        phase,
        Ref{Int32}(M.n),
        T[],
        M.csc.colptr,
        M.csc.rowval,
        M.perm,
        Ref(Int32(1)),
        M.iparm,
        M.msglvl,
        T[],
        T[],
        M.err,
        M.dparm,
    )
    @assert M.err[] == 0
end

struct PardisoSchurSolver{T} <: AbstractSchurSolver
    solver::PardisoSolver{T}
    nu::Int
    rhs::Vector{T}
    Sz::Vector{T}
    Si::Vector{Int32}
    Sj::Vector{Int32}
end

function _get_augmented_system(Ki, Bi, K0)
    nx, nu = size(Ki, 1), size(K0, 1)
    nnzK = nnz(Ki) + nnz(Bi) + nnz(K0)

    iK = zeros(Int32, nnzK)
    jK = zeros(Int32, nnzK)
    zK = zeros(Float64, nnzK)

    cnt = 1
    # Scan Ki
    for j in 1:nx
        for c in Ki.colptr[j]:Ki.colptr[j+1]-1
            i = Ki.rowval[c]
            v = Ki.nzval[c]
            iK[cnt] = i
            jK[cnt] = j
            # Add small elements on the diagonal for Pardiso
            zK[cnt] = if (i == j) && (v == 0.0)
                1e-8
            else
                v
            end
            cnt += 1
        end
    end
    # Scan Bi
    for j in 1:nu
        for c in Bi.colptr[j]:Bi.colptr[j+1]-1
            iK[cnt] = j + nx
            jK[cnt] = Bi.rowval[c]
            zK[cnt] = Bi.nzval[c]
            cnt += 1
        end
    end
    # Scan K0
    for j in 1:nu
        for c in K0.colptr[j]:K0.colptr[j+1]-1
            i = K0.rowval[c]
            v = K0.nzval[c]
            iK[cnt] = i + nx
            jK[cnt] = j + nx
            # Add small elements on the diagonal for Pardiso
            zK[cnt] = if (i == j) && (v == 0.0)
                1e-8
            else
                v
            end
            cnt += 1
        end
    end
    return sparse(iK, jK, zK, nx + nu, nx + nu)
end

function PardisoSchurSolver(
    Ki::SparseMatrixCSC{T,Int32},
    Bi::SparseMatrixCSC{T,Int32},
    K0::SparseMatrixCSC{T,Int32};
    options...,
) where {T}
    nki = size(Ki, 1)
    n_coupling = size(K0, 1)
    ntot = nki + n_coupling
    # Check consistency
    @assert size(K0, 2) == size(Bi, 2) == n_coupling
    @assert size(Ki, 2) == size(Bi, 1) == nki

    # Assemble full KKT system
    K = _get_augmented_system(Ki, Bi, K0)
    solver = PardisoSolver(K)

    rhs = zeros(T, ntot)

    # We assume the Schur-complement is dense.
    nnzS = div((n_coupling + 1) * n_coupling, 2)
    Si = zeros(Cint, n_coupling + 1)
    Sj = zeros(Cint, nnzS)
    Sz = zeros(T, nnzS)

    return PardisoSchurSolver{T}(solver, n_coupling, rhs, Sz, Si, Sj)
end

function solve!(rsol::PardisoSchurSolver{T}, x::AbstractVector) where {T}
    n = length(x)
    @assert rsol.solver.n == n + rsol.nu
    fill!(rsol.rhs, zero(T))
    copyto!(rsol.rhs, 1, x, 1, n)
    solve!(rsol.solver, rsol.rhs)
    copyto!(x, 1, rsol.rhs, 1, n)
    return x
end

function _transfer_values!(K, Ki, Bi, K0)
    nx = size(Ki, 1)
    nu = size(K0, 1)
    @assert size(K, 1) == size(K, 2) == nx + nu

    for j in 1:(nx+nu)
        for c in K.colptr[j]:K.colptr[j+1]-1
            i = K.rowval[c]
            if (i <= nx) && (j <= nx)
                K.nzval[c] = Ki[i, j]
            elseif (nx + 1 <= i <= nx + nu) && (j <= nx)
                K.nzval[c] = Bi[j, i-nx]
            elseif (nx + 1 <= i <= nx + nu) && (nx + 1 <= j <= nx + nu)
                K.nzval[c] = K0[i-nx, j-nx]
            else
                error("Wrong index")
            end
        end
    end
end

function schur_solve!(rsol::PardisoSchurSolver{T}, S, Ki, Bi, K0) where {T}
    solver = rsol.solver

    # Transfer values
    _transfer_values!(solver.csc, Ki, Bi, K0)

    # Update factorization
    solver.iparm[38] = rsol.nu
    factorize!(solver)

    # Get Schur-complement
    LibPardiso.pardiso_get_schur_d(
        solver.pt,
        solver.maxfct,
        solver.mnum,
        solver.mtype,
        rsol.Sz,
        rsol.Si,
        rsol.Sj,
    )

    # Unpack Schur-complement from CSR to dense matrix
    fill!(S, zero(T))
    for i in 1:rsol.nu
        for c in rsol.Si[i]:rsol.Si[i+1]-1
            j = rsol.Sj[c]
            v = rsol.Sz[c]
            S[i, j] += v
            if i != j
                S[j, i] += v
            end
        end
    end
end
