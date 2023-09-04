
#=
    Perform the KKT solve in parallel with MPI.

    ParallelKKTSysten acts as a thin wrapper (for dispatch) on top
    of MadNLP.SparseKKTSystem to implement the Schur decomposition
    of the KKT system.

    Ordering of the variables for the original problem:

    x  :=  (x1, x2, ..., xN, u)
    s  :=  (s1, s2, ..., sN)
    y  :=  (y1, y2, ..., yN)

    The block `id` views only the local variable `(x_id, s_id, y_id)`
    and the global control `u`.

=#

struct ParallelKKTSystem{T, VT, MT} <: MadNLP.AbstractReducedKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    inner::MadNLP.SparseKKTSystem{T, VT, SparseMatrixCSC{T, Int32}, MadNLP.ExactHessian{T, VT}}
    S::MT  # Schur-complement
    Ki::SparseMatrixCSC{T, Int}
    K0::SparseMatrixCSC{T, Int}
    Bi::SparseMatrixCSC{T, Int}
    n::Int
    m::Int
    n_ineq::Int
    n_coupling::Int
    # Mappings
    mapKi::Vector{Int}
    mapK0::Vector{Int}
    mapBi::Vector{Int}
    # MPI
    id::Int
    nblocks::Int
    pr_diag::VT
    du_diag::VT
    # Buffers
    # TODO
    _w1::VT
    _w2::VT
    _w3::VT
    _w4::VT
    _w5::VT
    _w6::VT
    rsol::MultipleRHSSolver{T}
    sparse_solver::MadNLPHSL.Ma57Solver
    comm::Union{MPI.Comm, Nothing}
end

function ParallelKKTSystem{T, VT, MT}(
    blk::BlockOPFModel,
    ind_cons=MadNLP.get_index_constraints(blk)
) where {T, VT, MT}

    id = blk.id
    nblocks = blk.nblocks
    comm = blk.comm

    nlp = blk.model
    SMT = SparseMatrixCSC{T, Int32}
    kkt = MadNLP.SparseKKTSystem{T, VT, SMT, MadNLP.ExactHessian{T, VT}}(nlp)

    # Local dimensions
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    ns = length(kkt.ind_ineq)
    n_coupling = blk.model.nlp.nu # TODO clean
    nki = n - n_coupling + ns + m

    # Global diagonal regularization
    pr_diag = VT(undef, nblocks * (n - n_coupling + ns) + n_coupling)
    fill!(pr_diag, zero(T))
    du_diag = VT(undef, nblocks * m)
    fill!(du_diag, zero(T))

    # Schur complement (we assume it is dense)
    S = zeros(Float64, n_coupling, n_coupling)

    # Reorder KKT system as Ki, K0 and Bi
    res = split_kkt_system(kkt.aug_com, n - n_coupling, n_coupling)
    K0 = sparse(res.i_K0, res.j_K0, convert.(Float64, res.mapK0))
    Ki = sparse(res.i_Ki, res.j_Ki, convert.(Float64, res.mapKi))
    Bi = sparse(res.i_Bi, res.j_Bi, convert.(Float64, res.mapBi), nki, n_coupling)

    nw = n + ns + m
    @assert size(Ki, 1) == size(kkt.aug_com, 1) - n_coupling

    # Get ordering in CSC matrices
    mapK0 = K0.nzval .|> Int
    mapKi = Ki.nzval .|> Int
    mapBi = Bi.nzval .|> Int

    # Buffers
    _w1 = zeros(nki)
    _w2 = zeros(nki)
    _w3 = zeros(n_coupling)
    _w4 = zeros(n_coupling)
    _w5 = zeros(m)
    _w6 = zeros(n+ns)


    nrhs = n_coupling
    sparse_solver = MadNLPHSL.Ma57Solver(Ki)
    rsol = MultipleRHSSolver(sparse_solver, nrhs)

    return ParallelKKTSystem{T, VT, MT}(
        kkt, S, Ki, K0, Bi, n, m, ns, n_coupling, mapKi, mapK0, mapBi,
        id, nblocks, pr_diag, du_diag, _w1, _w2, _w3, _w4, _w5, _w6, rsol, sparse_solver, comm,
    )
end

Base.size(kkt::ParallelKKTSystem, n::Int) = size(kkt.inner, n)
Base.size(kkt::ParallelKKTSystem) = size(kkt.inner)
MadNLP.num_variables(kkt::ParallelKKTSystem) = length(kkt.pr_diag)
MadNLP.get_hessian(kkt::ParallelKKTSystem) = MadNLP.get_hessian(kkt.inner)
MadNLP.get_jacobian(kkt::ParallelKKTSystem) = MadNLP.get_jacobian(kkt.inner)
MadNLP.is_reduced(::ParallelKKTSystem) = true
MadNLP.nnz_jacobian(kkt::ParallelKKTSystem) = MadNLP.nnz_jacobian(kkt.inner)
MadNLP.get_kkt(kkt::ParallelKKTSystem) = kkt.S

MadNLP.initialize!(kkt::ParallelKKTSystem) = MadNLP.initialize!(kkt.inner)
MadNLP.get_raw_jacobian(kkt::ParallelKKTSystem) = MadNLP.get_raw_jacobian(kkt.inner)

function MadNLP.set_jacobian_scaling!(kkt::ParallelKKTSystem{T,VT,MT}, constraint_scaling::AbstractVector) where {T,VT,MT}
    m = kkt.m
    shift_c = kkt.id * m
    _c = view(constraint_scaling, shift_c+1:shift_c+m)
    MadNLP.set_jacobian_scaling!(kkt.inner, _c)
    return
end

function MadNLP.mul!(y::AbstractVector, kkt::ParallelKKTSystem, x::AbstractVector)
    mul!(y, kkt.inner, x)
end

function MadNLP.jtprod!(
    y_h::AbstractVector,
    kkt::ParallelKKTSystem{T, VT, MT},
    x_h::AbstractVector,
) where {T, VT, MT}
    fill!(y_h, 0.0)
    nu = kkt.n_coupling
    nx = kkt.n - nu
    m = kkt.m
    ns = kkt.n_ineq

    shift_x = kkt.id * nx
    shift_u = kkt.nblocks * nx
    shift_s = kkt.nblocks * nx + nu + kkt.id * ns
    shift_c = kkt.id * m

    _x = kkt._w5
    _y = kkt._w6

    # Load local variable
    copyto!(_x, 1, x_h, shift_c+1, m)

    MadNLP.jtprod!(_y, kkt.inner, _x)

    # Unpack result
    copyto!(y_h, shift_x+1, _y, 1, nx)
    copyto!(y_h, shift_u+1, _y, nx+1, nu)
    copyto!(y_h, shift_s+1, _y, nx+nu+1, ns)

    # Sum contributions
    comm_sum!(y_h, kkt.comm)
end

MadNLP.compress_jacobian!(kkt::ParallelKKTSystem) = MadNLP.compress_jacobian!(kkt.inner)
MadNLP.compress_hessian!(kkt::ParallelKKTSystem) = MadNLP.compress_hessian!(kkt.inner)

function MadNLP.build_kkt!(kkt::ParallelKKTSystem)
    MadNLP.build_kkt!(kkt.inner)

    # Update matrices
    new_vals = kkt.inner.aug_com.nzval
    kkt.Ki.nzval .= new_vals[kkt.mapKi]
    kkt.K0.nzval .= new_vals[kkt.mapK0]
    kkt.Bi.nzval .= new_vals[kkt.mapBi]

    # Update factorization
    copyto!(kkt.sparse_solver.csc.nzval, kkt.Ki.nzval)
    MadNLP.factorize!(kkt.sparse_solver)

    kkt.S .= kkt.K0                         #  S = K₀
    KBi = solve!(kkt.rsol, kkt.Bi)          #  KBi = Kᵢ⁻¹ Bᵢ
    mul!(kkt.S, kkt.Bi', KBi, -1.0, 1.0)    #  S += - Bᵢᵀ Kᵢ⁻¹ Bᵢ

    # Assemble Schur complement (reduction) on all processes
    comm_sum!(kkt.S, kkt.comm)
    return
end

function MadNLP.solve_refine_wrapper!(
    solver::MadNLP.MadNLPSolver{T, <:ParallelKKTSystem{T,VT,MT}},
    x_r::MadNLP.AbstractKKTVector, b_r::MadNLP.AbstractKKTVector,
) where {T, VT, MT}
    kkt = solver.kkt
    comm = kkt.comm
    # MPI
    id = solver.kkt.id
    nblocks = solver.kkt.nblocks
    m = kkt.m
    nu = solver.kkt.n_coupling
    nx = kkt.n - nu
    ns = kkt.n_ineq

    x_h = MadNLP.full(x_r)
    b_h = MadNLP.full(b_r)

    # Transfer
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns
    shift_y = nblocks * (nx + ns) + nu + id * m

    _x = solver.kkt._w1
    _b = solver.kkt._w2
    _xu = solver.kkt._w3
    _bu = solver.kkt._w4

    # Initiate RHS
    copyto!(_b,       1, b_h, shift_x + 1, nx)
    copyto!(_b,    nx+1, b_h, shift_s + 1, ns)
    copyto!(_b, nx+ns+1, b_h, shift_y + 1, m)
    copyto!(_bu, 1, b_h, shift_u + 1, nu)

    # TODO
    # MadNLP.fixed_variable_treatment_vec!(_b, solver.ind_fixed)

    # Step 1. Assemble RHS for Schur-complement
    _bu ./= nblocks
    _x .= _b
    MadNLP.solve!(solver.kkt.sparse_solver, _x)
    mul!(_bu, solver.kkt.Bi', _x, -1.0, 1.0)
    _xu .= _bu
    # Sum contributions across all processes
    comm_sum!(_xu, comm)

    # Step 2. Evaluate direction w.r.t. u
    solver.cnt.linear_solver_time += @elapsed begin
        MadNLP.solve!(solver.linear_solver, _xu)
    end

    # Step 3. Individual contribution
    _x .= _b
    mul!(_x, kkt.Bi, _xu, -1.0, 1.0)
    MadNLP.solve!(solver.kkt.sparse_solver, _x)

    # TODO
    # MadNLP.fixed_variable_treatment_vec!(x, ips.ind_fixed)

    fill!(x_h, zero(T))
    # / x
    copyto!(x_h, shift_x + 1, _x,       1, nx)
    copyto!(x_h, shift_s + 1, _x,    nx+1, ns)
    copyto!(x_h, shift_y + 1, _x, nx+ns+1, m )
    # / u
    copyto!(x_h, shift_u + 1, _xu, 1, nu)

    # Scale coupling's direction by number of processes before summing it
    x_h[shift_u+1:shift_u+nu] ./= nblocks

    comm_sum!(x_h, comm)

    return true
end

# set_aug_diagonal
function _synchronize_regularization!(kkt::ParallelKKTSystem)
    nblocks = kkt.nblocks
    id = kkt.id
    nx = kkt.n - kkt.n_coupling
    nu = kkt.n_coupling
    ns = kkt.n_ineq
    m = kkt.m

    # Sync-up with internal KKT
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns

    copyto!(kkt.inner.pr_diag,       1, kkt.pr_diag, shift_x+1, nx)
    copyto!(kkt.inner.pr_diag,    1+nx, kkt.pr_diag, shift_u+1, nu)
    copyto!(kkt.inner.pr_diag, 1+nx+nu, kkt.pr_diag, shift_s+1, ns)

    # We dispatch the regularization evently across the different processes
    kkt.inner.pr_diag[1+nx:nx+nu] ./= kkt.nblocks

    shift_c = id * m
    copyto!(kkt.inner.du_diag, 1, kkt.du_diag, shift_c+1, m)
    return
end

function MadNLP.set_aug_diagonal!(kkt::ParallelKKTSystem, ips::MadNLP.MadNLPSolver)
    # Load data
    x = MadNLP.full(ips.x)
    xl = MadNLP.full(ips.xl)
    xu = MadNLP.full(ips.xu)
    zl = MadNLP.full(ips.zl)
    zu = MadNLP.full(ips.zu)
    _pr_diag = zeros(length(kkt.pr_diag))
    # Global regularization
    _pr_diag .= zl./(x.-xl) .+ zu./(xu.-x)
    copyto!(kkt.pr_diag, _pr_diag)
    fill!(kkt.du_diag, 0.0)
    _synchronize_regularization!(kkt)
end

function MadNLP.regularize_diagonal!(kkt::ParallelKKTSystem, primal, dual)
    kkt.pr_diag .+= primal
    kkt.du_diag .= .-dual
    _synchronize_regularization!(kkt)
end

function MadNLP.is_inertia_correct(kkt::ParallelKKTSystem, num_pos, num_zero, num_neg)
    return num_pos == kkt.n_coupling
end
