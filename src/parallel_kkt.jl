
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

struct ParallelKKTSystem{T,VT,MT,LS} <:
       MadNLP.AbstractReducedKKTSystem{T,VT,MT,MadNLP.ExactHessian{T,VT}}
    local_kkt::AugmentedKKTMatrix{T, VT, SparseMatrixCSC{T, Int32}}
    # Communication
    S::MT  # Schur-complement
    Ki::SparseMatrixCSC{T,Int}
    K0::SparseMatrixCSC{T,Int}
    Bi::SparseMatrixCSC{T,Int}
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
    # Regularization
    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::Vector{T}
    u_diag::Vector{T}
    l_lower::Vector{T}
    u_lower::Vector{T}
    # Info
    ind_ineq::Vector{Int}
    ind_lb::Vector{Int}
    ind_ub::Vector{Int}
    # Buffers
    _w::VT
    _v::VT
    _w1::VT
    _w2::VT
    _w3::VT
    _w4::VT
    _w5::VT
    _w6::VT
    # Linear solvers
    rsol::AbstractSchurSolver
    linear_solver::LS
    comm::Union{MPI.Comm,Nothing}
    etc::Dict{Symbol,Any}
end

function MadNLP.create_kkt_system(
    ::Type{ParallelKKTSystem{T,VT,MT}},
    cb::MadNLP.SparseCallback{T, VT},
    opt,
    opt_linear_solver,
    cnt,
    ind_cons;
) where {T,VT,MT}
    SMT = SparseMatrixCSC{T,Int32}

    blk = cb.nlp

    id = blk.id
    nblocks = blk.nblocks
    comm = blk.comm

    nlp = blk.model

    inner_cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp,
        opt,
    )
    _ind_cons = MadNLP.get_index_constraints(
        nlp,
        opt.fixed_variable_treatment,
        opt.equality_treatment,
    )

    local_aug_kkt = AugmentedKKTMatrix(inner_cb, _ind_cons)

    # Local dimensions
    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    ns = length(_ind_cons.ind_ineq)
    n_coupling = blk.model.nlp.nu # TODO clean
    nki = n - n_coupling + ns + m
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)

    # Global diagonal regularization
    nvars = nblocks * (n - n_coupling + ns) + n_coupling
    reg = zeros(nvars)
    pr_diag = VT(undef, nvars)
    fill!(pr_diag, zero(T))
    du_diag = VT(undef, nblocks * m)
    fill!(du_diag, zero(T))

    l_diag = zeros(nlb)
    u_diag = zeros(nub)
    l_lower = zeros(nlb)
    u_lower = zeros(nub)

    # Schur complement (we assume it is dense)
    S = zeros(Float64, n_coupling, n_coupling)

    # Reorder KKT system as Ki, K0 and Bi
    res = split_kkt_system(local_aug_kkt.aug_com, n - n_coupling, n_coupling)
    K0 = sparse(res.i_K0, res.j_K0, convert.(Float64, res.mapK0))
    Ki = sparse(res.i_Ki, res.j_Ki, convert.(Float64, res.mapKi))
    Bi = sparse(res.i_Bi, res.j_Bi, convert.(Float64, res.mapBi), nki, n_coupling)

    @assert size(Ki, 1) == size(local_aug_kkt.aug_com, 1) - n_coupling

    # Get ordering in CSC matrices
    mapK0 = K0.nzval .|> Int
    mapKi = Ki.nzval .|> Int
    mapBi = Bi.nzval .|> Int

    # Buffers
    nw = n + ns + m
    _w = zeros(nw)
    _v = zeros(nw)
    _w1 = MadNLP._madnlp_unsafe_wrap(_w, nki)
    _w2 = MadNLP._madnlp_unsafe_wrap(_v, nki)
    _w3 = MadNLP._madnlp_unsafe_wrap(_w, n_coupling, nki + 1)
    _w4 = MadNLP._madnlp_unsafe_wrap(_v, n_coupling, nki + 1)
    _w5 = zeros(m)
    _w6 = zeros(n + ns)

    rsol = Ma57SchurSolver(Ki, Bi, K0)
    # rsol = PardisoSchurSolver(Ki, Bi, K0; linear_solver_options...)
    linear_solver = MadNLP.LapackCPUSolver(S)

    etc = Dict{Symbol,Any}()
    etc[:reduction] = 0.0
    etc[:comm] = 0.0
    etc[:backsolve] = 0.0

    return ParallelKKTSystem{T,VT,MT,typeof(linear_solver)}(
        local_aug_kkt,
        S,
        Ki,
        K0,
        Bi,
        n,
        m,
        ns,
        n_coupling,
        mapKi,
        mapK0,
        mapBi,
        id,
        nblocks,
        reg,
        pr_diag,
        du_diag,
        l_diag,
        u_diag,
        l_lower,
        u_lower,
        ind_cons.ind_ineq,
        ind_cons.ind_lb,
        ind_cons.ind_ub,
        _w,
        _v,
        _w1,
        _w2,
        _w3,
        _w4,
        _w5,
        _w6,
        rsol,
        linear_solver,
        comm,
        etc,
    )
end

Base.size(kkt::ParallelKKTSystem, n::Int) = length(kkt.pr_diag) + length(kkt.du_diag)
function Base.size(kkt::ParallelKKTSystem)
    m = length(kkt.pr_diag) + length(kkt.du_diag)
    return (m, m)
end

MadNLP.num_variables(kkt::ParallelKKTSystem) = length(kkt.pr_diag)
MadNLP.get_hessian(kkt::ParallelKKTSystem) = kkt.local_kkt.hess
MadNLP.get_jacobian(kkt::ParallelKKTSystem) = kkt.local_kkt.jac_callback
MadNLP.is_reduced(::ParallelKKTSystem) = true
MadNLP.nnz_jacobian(kkt::ParallelKKTSystem) = nnz(kkt.local_kkt.jac_raw)
MadNLP.get_kkt(kkt::ParallelKKTSystem) = kkt.S

function MadNLP.initialize!(kkt::ParallelKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    # Init local kkt values
    local_kkt = kkt.local_kkt
    fill!(local_kkt.pr_diag, 1.0)
    fill!(nonzeros(local_kkt.hess_com), 0.0)
end

function _mul_expanded!(y::AbstractVector, kkt::ParallelKKTSystem, x::AbstractVector, alpha, beta)
    nu = kkt.n_coupling
    nx = kkt.n - nu
    m = kkt.m
    ns = kkt.n_ineq

    local_kkt = kkt.local_kkt

    shift_x = kkt.id * nx
    shift_u = kkt.nblocks * nx
    shift_s = kkt.nblocks * nx + nu + kkt.id * ns
    shift_y = kkt.nblocks * (nx + ns) + nu + kkt.id * m

    # Local buffers
    _y = kkt._w
    _x = kkt._v

    # Copy values to local arrays.
    copyto!(_x, 1, x, shift_x + 1, nx)
    copyto!(_x, 1 + nx, x, shift_u + 1, nu)
    copyto!(_x, nx + nu + 1, x, shift_s + 1, ns)
    copyto!(_x, nx + nu + ns + 1, x, shift_y + 1, m)

    copyto!(_y, 1, y, shift_x + 1, nx)
    copyto!(_y, 1 + nx, y, shift_u + 1, nu)
    copyto!(_y, nx + nu + 1, y, shift_s + 1, ns)
    copyto!(_y, nx + nu + ns + 1, y, shift_y + 1, m)

    _xp = view(_x, 1:nx+nu+ns)
    _xl = view(_x, nx+nu+ns+1:nx+nu+ns+m)

    _yp = view(_y, 1:nx+nu+ns)
    _yl = view(_y, nx+nu+ns+1:nx+nu+ns+m)

    mul!(_yp, Symmetric(local_kkt.hess_com, :L), _xp, alpha, beta)
    mul!(_yp, local_kkt.jac_com', _xl, alpha, 1.0)
    mul!(_yl, local_kkt.jac_com,  _xp, alpha, beta)

    # Copy values back to global array.
    copyto!(y, shift_x + 1, _y, 1, nx)
    copyto!(y, shift_u + 1, _y, nx + 1, nu)
    copyto!(y, shift_s + 1, _y, nx + nu + 1, ns)
    copyto!(y, shift_y + 1, _y, nx + nu + ns + 1, m)

    tic = comm_walltime(kkt.comm)
    comm_sum!(y, kkt.comm)
    kkt.etc[:comm] += comm_walltime(kkt.comm) - tic
    return
end

function MadNLP.mul!(
    y::MadNLP.AbstractKKTVector,
    kkt::ParallelKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha,
    beta,
)
    _mul_expanded!(MadNLP.full(y), kkt, MadNLP.full(x), alpha, beta)
    MadNLP._kktmul!(y, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return y
end

function MadNLP.jtprod!(
    y_h::AbstractVector,
    kkt::ParallelKKTSystem{T,VT,MT},
    x_h::AbstractVector,
) where {T,VT,MT}
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
    copyto!(_x, 1, x_h, shift_c + 1, m)

    mul!(_y, kkt.local_kkt.jac_com', _x)

    # Unpack result
    copyto!(y_h, shift_x + 1, _y, 1, nx)
    copyto!(y_h, shift_u + 1, _y, nx + 1, nu)
    copyto!(y_h, shift_s + 1, _y, nx + nu + 1, ns)

    # Sum contributions
    tic = comm_walltime(kkt.comm)
    comm_sum!(y_h, kkt.comm)
    kkt.etc[:comm] += comm_walltime(kkt.comm) - tic
    return
end

function MadNLP.compress_jacobian!(kkt::ParallelKKTSystem)
    ns = length(kkt.ind_ineq)
    kkt.local_kkt.jac[end-ns+1:end] .= -1.0
    MadNLP.transfer!(kkt.local_kkt.jac_com, kkt.local_kkt.jac_raw, kkt.local_kkt.jac_csc_map)
end

function MadNLP.compress_hessian!(kkt::ParallelKKTSystem)
    MadNLP.transfer!(kkt.local_kkt.hess_com, kkt.local_kkt.hess_raw, kkt.local_kkt.hess_csc_map)
end

function assemble!(kkt::ParallelKKTSystem)
    return schur_solve!(kkt.rsol, kkt.S, kkt.Ki, kkt.Bi, kkt.K0)
end

function MadNLP.build_kkt!(kkt::ParallelKKTSystem)
    local_kkt = kkt.local_kkt
    MadNLP.transfer!(local_kkt.aug_com, local_kkt.aug_raw, local_kkt.aug_csc_map)

    # Copy values back into (Ki, Bi, K0)
    new_vals = local_kkt.aug_com.nzval
    kkt.Ki.nzval .= new_vals[kkt.mapKi]
    kkt.K0.nzval .= new_vals[kkt.mapK0]
    kkt.Bi.nzval .= new_vals[kkt.mapBi]

    # Assemble Schur-complement
    kkt.etc[:reduction] += @elapsed begin
        assemble!(kkt)
    end

    # Assemble Schur complement (reduction) on all processes
    tic = comm_walltime(kkt.comm)
    comm_sum!(kkt.S, kkt.comm)
    kkt.etc[:comm] += comm_walltime(kkt.comm) - tic
    symmetrize!(kkt.S)

    return
end

function MadNLP.solve!(kkt::ParallelKKTSystem, w::MadNLP.AbstractKKTVector)
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)
    comm = kkt.comm
    # MPI
    id = kkt.id
    nblocks = kkt.nblocks
    m = kkt.m
    nu = kkt.n_coupling
    nx = kkt.n - nu
    ns = kkt.n_ineq

    # Transfer
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns
    shift_y = nblocks * (nx + ns) + nu + id * m

    x_full = MadNLP.full(w)

    _x = kkt._w1
    _xu = kkt._w3
    _b = kkt._w2

    # Initiate RHS
    copyto!(_x, 1, x_full, shift_x + 1, nx)
    copyto!(_x, nx + 1, x_full, shift_s + 1, ns)
    copyto!(_x, nx + ns + 1, x_full, shift_y + 1, m)
    copyto!(_xu, 1, x_full, shift_u + 1, nu)

    # Save initial RHS
    _b .= _x

    # Step 1. Assemble RHS for Schur-complement
    _xu ./= nblocks
    kkt.etc[:backsolve] += @elapsed begin
        solve!(kkt.rsol, _x)
    end
    mul!(_xu, kkt.Bi', _x, -1.0, 1.0)

    tic = comm_walltime(kkt.comm)
    comm_sum!(_xu, comm)
    kkt.etc[:comm] += comm_walltime(kkt.comm) - tic

    # Step 2. Evaluate direction w.r.t. u
    MadNLP.solve!(kkt.linear_solver, _xu)

    # Step 3. Individual contribution
    _x .= _b
    mul!(_x, kkt.Bi, _xu, -1.0, 1.0)
    kkt.etc[:backsolve] += @elapsed begin
        solve!(kkt.rsol, _x)
    end

    # / x
    copyto!(x_full, shift_x + 1, _x, 1, nx)
    copyto!(x_full, shift_s + 1, _x, nx + 1, ns)
    copyto!(x_full, shift_y + 1, _x, nx + ns + 1, m)
    # / u
    copyto!(x_full, shift_u + 1, _xu, 1, nu)

    # Scale coupling's direction by number of processes before summing it
    x_full[shift_u+1:shift_u+nu] ./= nblocks

    tic = comm_walltime(kkt.comm)
    comm_sum!(x_full, comm)
    kkt.etc[:comm] += comm_walltime(kkt.comm) - tic

    MadNLP.finish_aug_solve!(kkt, w)
    return w
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

    copyto!(kkt.local_kkt.pr_diag, 1, kkt.pr_diag, shift_x + 1, nx)
    copyto!(kkt.local_kkt.pr_diag, 1 + nx, kkt.pr_diag, shift_u + 1, nu)
    copyto!(kkt.local_kkt.pr_diag, 1 + nx + nu, kkt.pr_diag, shift_s + 1, ns)

    # We dispatch the regularization evently across the different processes
    kkt.local_kkt.pr_diag[1+nx:nx+nu] ./= kkt.nblocks

    shift_c = id * m
    copyto!(kkt.local_kkt.du_diag, 1, kkt.du_diag, shift_c + 1, m)
    return
end

function MadNLP.set_aug_diagonal!(kkt::ParallelKKTSystem{T}, solver::MadNLP.MadNLPSolver) where T
    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)
    MadNLP._set_aug_diagonal!(kkt)
    return
end

function MadNLP._set_aug_diagonal!(kkt::ParallelKKTSystem)
    copyto!(kkt.pr_diag, kkt.reg)
    kkt.pr_diag[kkt.ind_lb] .-= kkt.l_lower ./ kkt.l_diag
    kkt.pr_diag[kkt.ind_ub] .-= kkt.u_lower ./ kkt.u_diag
    _synchronize_regularization!(kkt)
    return
end

function MadNLP.regularize_diagonal!(kkt::ParallelKKTSystem, primal, dual)
    kkt.pr_diag .+= primal
    kkt.du_diag .= .-dual
    _synchronize_regularization!(kkt)
    return
end

function MadNLP.is_inertia_correct(kkt::ParallelKKTSystem, num_pos, num_zero, num_neg)
    return num_pos == kkt.n_coupling
end
