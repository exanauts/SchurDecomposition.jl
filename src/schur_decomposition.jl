
#=
    Perform the reduction in parallel with MPI.

    SchurKKTSystem acts as a thin wrapper (for dispatch) on top
    of Argos.BieglerKKTSystem to perform the parallel reduction.

    We exploit directly the structure of the optimal power flow,
    as in this case the reduction is equivalent to the Schur complement
    decomposition.

    Ordering of the variables for the original problem:

    x  :=  (x1, x2, ..., xN, u)
    s  :=  (s1, s2, ..., sN)
    y  :=  (y1, y2, ..., yN)

=#

struct SchurKKTSystem{T,VI,VT,MT,LS} <:
       MadNLP.AbstractReducedKKTSystem{T,VT,MT,MadNLP.ExactHessian{T,VT}}
    inner::Argos.BieglerKKTSystem{T,VI,VT,MT}
    n::Int
    m::Int
    n_ineq::Int
    n_coupling::Int
    id::Int
    nblocks::Int
    reg::Vector{T}
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
    _w1::VT
    _w2::VT
    _w3::VT
    _w4::VT
    _w5::VT
    linear_solver::LS
    comm::Union{MPI.Comm,Nothing}
    etc::Dict{Symbol,Any}
end

function MadNLP.create_kkt_system(
    ::Type{SchurKKTSystem{T, VI, VT, MT}},
    cb::MadNLP.SparseCallback{T, Vector{T}},
    opt,
    opt_linear_solver,
    cnt,
    ind_cons;
    max_batches=256,
) where {T, VI, VT, MT}
    blk = cb.nlp::BlockOPFModel

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

    kkt = MadNLP.create_kkt_system(
        Argos.BieglerKKTSystem{T,VI,VT,MT},
        inner_cb,
        opt, opt_linear_solver, cnt, _ind_cons;
        max_batches=max_batches,
    )
    nx = kkt.nx
    nu = kkt.nu
    nlb, nub = length(ind_cons.ind_lb), length(ind_cons.ind_ub)

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    ns = length(kkt.ind_ineq)

    pr_diag = VT(undef, nblocks * (nx + ns) + nu)
    du_diag = VT(undef, nblocks * m)
    fill!(pr_diag, zero(T))
    fill!(du_diag, zero(T))

    # Buffers
    _w1 = zeros(T, n)
    _w2 = zeros(T, m)
    _w3 = zeros(T, n + ns)
    _w4 = zeros(T, n + ns + m)
    _w5 = zeros(T, n + ns + m)
    reg = zeros(nblocks * (nx + ns) + nu)

    l_diag = zeros(nlb)
    u_diag = zeros(nub)
    l_lower = zeros(nlb)
    u_lower = zeros(nub)

    linear_solver = kkt.linear_solver

    # Timers
    etc = Dict{Symbol,Any}()
    etc[:reduction] = 0.0
    etc[:comm] = 0.0
    etc[:backsolve] = 0.0

    return SchurKKTSystem{T,VI,VT,MT,typeof(linear_solver)}(
        kkt,
        n, m, ns, nu,
        id,
        nblocks,
        reg, pr_diag, du_diag,
        l_diag, u_diag, l_lower, u_lower,
        ind_cons.ind_ineq, ind_cons.ind_lb, ind_cons.ind_ub,
        _w1,
        _w2,
        _w3,
        _w4,
        _w5,
        linear_solver,
        comm,
        etc,
    )
end

Base.size(kkt::SchurKKTSystem, n::Int) = length(kkt.pr_diag) + length(kkt.du_diag)
function Base.size(kkt::SchurKKTSystem)
    m = length(kkt.pr_diag) + length(kkt.du_diag)
    return (m, m)
end
MadNLP.num_variables(kkt::SchurKKTSystem) = kkt.inner.nu
MadNLP.get_hessian(kkt::SchurKKTSystem) = MadNLP.get_hessian(kkt.inner)
MadNLP.get_jacobian(kkt::SchurKKTSystem) = MadNLP.get_jacobian(kkt.inner)
MadNLP.is_reduced(::SchurKKTSystem) = true
MadNLP.nnz_jacobian(kkt::SchurKKTSystem) = MadNLP.nnz_jacobian(kkt.inner)
MadNLP.get_kkt(kkt::SchurKKTSystem) = kkt.inner.aug_com

function MadNLP.initialize!(kkt::SchurKKTSystem)
    fill!(kkt.reg, 1.0)
    fill!(kkt.pr_diag, 1.0)
    fill!(kkt.du_diag, 0.0)
    fill!(kkt.l_lower, 0.0)
    fill!(kkt.u_lower, 0.0)
    fill!(kkt.l_diag, 1.0)
    fill!(kkt.u_diag, 1.0)
    fill!(kkt.inner.pr_diag, 1.0)
end
MadNLP.get_raw_jacobian(kkt::SchurKKTSystem) = MadNLP.get_raw_jacobian(kkt.inner)

# TODO: mul expanded
function mul!(
    y::MadNLP.AbstractKKTVector,
    kkt::SchurKKTSystem,
    x::MadNLP.AbstractKKTVector,
    alpha,
    beta,
)
    m = kkt.m
    nu = kkt.n_coupling
    nx = kkt.inner.nx
    ns = kkt.n_ineq

    _y = kkt._w4
    _x = kkt._w5

    shift_x = kkt.id * nx
    shift_u = kkt.nblocks * nx
    shift_s = kkt.nblocks * nx + nu + kkt.id * ns
    shift_y = kkt.nblocks * (nx + ns) + nu + kkt.id * m

    copyto!(_x, 1, MadNLP.full(x), shift_x + 1, nx)
    copyto!(_x, 1 + nx, MadNLP.full(x), shift_u + 1, nu)
    copyto!(_x, nx + nu + 1, MadNLP.full(x), shift_s + 1, ns)
    copyto!(_x, nx + nu + ns + 1, MadNLP.full(x), shift_y + 1, m)

    copyto!(_y, 1, MadNLP.full(y), shift_x + 1, nx)
    copyto!(_y, 1 + nx, MadNLP.full(y), shift_u + 1, nu)
    copyto!(_y, nx + nu + 1, MadNLP.full(y), shift_s + 1, ns)
    copyto!(_y, nx + nu + ns + 1, MadNLP.full(y), shift_y + 1, m)

    Argos._mul_expanded!(_y, kkt.inner, _x, alpha, beta)

    copyto!(MadNLP.full(y), shift_x + 1, _y, 1, nx)
    copyto!(MadNLP.full(y), shift_u + 1, _y, nx + 1, nu)
    copyto!(MadNLP.full(y), shift_s + 1, _y, nx + nu + 1, ns)
    copyto!(MadNLP.full(y), shift_y + 1, _y, nx + nu + ns + 1, m)

    MadNLP._kktmul!(y, x, kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return y
end

function MadNLP.jtprod!(
    y_h::AbstractVector,
    kkt::SchurKKTSystem{T,VI,VT,MT},
    x_h::AbstractVector,
) where {T,VI,VT,MT}
    fill!(y_h, 0.0)
    nx = kkt.inner.nx
    nu = kkt.inner.nu
    m = size(kkt.inner.J, 1)
    ns = length(kkt.inner.ind_ineq)
    shift_x = kkt.id * nx
    shift_u = kkt.nblocks * nx
    shift_s = kkt.nblocks * nx + nu + kkt.id * ns
    shift_c = kkt.id * m

    _x = kkt._w2
    _y = kkt._w3

    # Load local variable
    copyto!(_x, 1, x_h, shift_c + 1, m)

    MadNLP.jtprod!(_y, kkt.inner, _x)

    # Unpack result
    copyto!(y_h, shift_x + 1, _y, 1, nx)
    copyto!(y_h, shift_u + 1, _y, nx + 1, nu)
    copyto!(y_h, shift_s + 1, _y, nx + nu + 1, ns)

    # Sum contributions
    tic = comm_walltime(kkt.comm)
    comm_sum!(y_h, kkt.comm)
    return kkt.etc[:comm] += comm_walltime(kkt.comm) - tic
end

MadNLP.compress_jacobian!(kkt::SchurKKTSystem) = MadNLP.compress_jacobian!(kkt.inner)
MadNLP.compress_hessian!(kkt::SchurKKTSystem) = MadNLP.compress_hessian!(kkt.inner)

function MadNLP.build_kkt!(kkt::SchurKKTSystem)
    kkt.etc[:reduction] += @elapsed begin
        MadNLP.build_kkt!(kkt.inner)
    end
    # Assemble Schur complement (reduction) on all processes
    tic = comm_walltime(kkt.comm)
    comm_sum!(kkt.inner.aug_com, kkt.comm)
    return kkt.etc[:comm] += comm_walltime(kkt.comm) - tic
end

function MadNLP.solve!(
    kkt::SchurKKTSystem{T,VI,VT,MT},
    w::MadNLP.AbstractKKTVector,
) where {T,VI,VT,MT}
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)
    # MPI
    id = kkt.id
    nblocks = kkt.nblocks
    comm = kkt.comm

    inner_kkt = kkt.inner
    # Problem's dimension
    m = kkt.m  # constraints
    nx, nu = inner_kkt.nx, inner_kkt.nu # state and control
    ns = kkt.n_ineq
    n = nx + nu + ns # local number of variables, replace solver.n

    x_full = MadNLP.full(w)
    # Transfer
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns
    shift_y = nblocks * (nx + ns) + nu + id * m

    # Local solve
    _x = kkt._w4

    copyto!(_x,                1, x_full, shift_x + 1, nx)
    copyto!(_x,           nx + 1, x_full, shift_u + 1, nu)
    copyto!(_x,      nx + nu + 1, x_full, shift_s + 1, ns)
    copyto!(_x, nx + nu + ns + 1, x_full, shift_y + 1, m)

    # Transfer data to device
    x = Argos._load_buffer(inner_kkt, _x, :kkt_x)::VT

    # Buffers
    jv = inner_kkt._wxu1
    tx = view(jv, 1:nx)
    tu = view(jv, nx+1:nx+nu)
    vj = inner_kkt._wj1
    sx1 = inner_kkt._wx1
    sx2 = inner_kkt._wx2
    # Views for Hessian-vector multiplication
    kv = inner_kkt._wxu2
    kh = inner_kkt._wxu3
    kvx = view(kv, 1:nx)
    kvu = view(kv, nx+1:nx+nu)
    khx = view(kh, 1:nx)
    khu = view(kh, nx+1:nx+nu)

    # Gₓ⁻¹
    Gxi = inner_kkt.G_fac
    Gx = inner_kkt.Gx
    Gu = inner_kkt.Gu
    K = inner_kkt.K
    Σₛ = view(inner_kkt.pr_diag, nx+nu+1:nx+nu+ns)
    vs = inner_kkt._wj2

    # Decompose solution
    dxu = view(x, 1:nx+nu)
    dx = view(x, 1:nx)           # / state
    du = view(x, 1+nx:nx+nu)     # / control
    ds = view(x, 1+nx+nu:n)      # / slack
    dλ = view(x, n+1:n+nx)       # / equality cons
    dy = view(x, n+nx+1:n+m)     # / inequality cons

    du ./= nblocks
    # Reduction (1) --- Condensed
    vj .= (Σₛ .* dy .+ ds)                # v = (Σₛ r₅ + α r₃)
    mul!(jv, inner_kkt.A', vj, one(T), zero(T)) # jᵥ = Aᵀ v
    jv .+= dxu                            # r₁₂ - Aᵀv
    # Reduction (2) --- Biegler
    sx1 .= dλ                             # r₄
    ldiv!(Gxi, sx1)                       # Gₓ⁻¹ r₄

    sx2 .= tx                             # tx = jv[1:nx]
    kvx .= sx1 ; kvu .= zero(T)
    mul!(kh, K, kv)                       # [Kₓₓ Gₓ⁻¹ r₄ ; Kᵤₓ Gₓ⁻¹ r₄ ]
    sx2 .= khx .- tx                      # sₓ = Kₓₓ Gₓ⁻¹ r₄ - tₓ
    ldiv!(Gxi', sx2)                      # Gₓ⁻ᵀ sₓ
    mul!(tu, Gu', sx2, one(T), one(T))    # tᵤ = tᵤ + Gᵤᵀ Gₓ⁻ᵀ sₓ
    axpy!(-one(T), khu, tu)               # tᵤ = tᵤ - Kᵤₓ Gₓ⁻¹ r₄

    du .= tu
    tic = comm_walltime(comm)
    comm_sum!(du, comm)
    kkt.etc[:comm] += comm_walltime(comm) - tic

    MadNLP.solve!(kkt.linear_solver, du)

    # (1) Extract Biegler
    dx .= dλ                              # r₄
    mul!(dx, Gu, du, -one(T), one(T))     # r₄ - Gᵤ dᵤ
    ldiv!(Gxi, dx)                        # dₓ = Gₓ⁻¹ (r₄ - Gᵤ dᵤ)
    dλ .= tx                              # tₓ
    mul!(kh, K, dxu)                      # Kₓₓ dₓ + Kₓᵤ dᵤ
    axpy!(-one(T), khx, dλ)               # tₓ - Kₓₓ dₓ - Kₓᵤ dᵤ

    ldiv!(Gxi', dλ)                       # dₗ = Gₓ⁻ᵀ(tₓ - Kₓₓ dₓ + Kₓᵤ dᵤ)

    # (2) Extract Condensed
    mul!(vj, inner_kkt.A, dxu)                  # Aₓ dₓ + Aᵤ dᵤ
    copyto!(vs, ds)
    ds .= (vj .- dy)
    dy .= Σₛ .* ds .- vs

    copyto!(x_full, shift_x + 1, x,                1, nx)
    copyto!(x_full, shift_u + 1, x,           nx + 1, nu)
    copyto!(x_full, shift_s + 1, x,      nx + nu + 1, ns)
    copyto!(x_full, shift_y + 1, x, nx + nu + ns + 1, m)

    # Scale coupling's direction by number of processes
    x_full[shift_u+1:shift_u+nu] ./= nblocks

    tic = comm_walltime(comm)
    comm_sum!(x_full, comm)
    kkt.etc[:comm] += comm_walltime(comm) - tic

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

# set_aug_diagonal
function _synchronize_regularization!(kkt::SchurKKTSystem)
    nblocks = kkt.nblocks
    id = kkt.id
    nx = kkt.inner.nx
    nu = kkt.inner.nu
    ns = length(kkt.inner.ind_ineq)
    m = size(kkt.inner.J, 1)

    # Sync-up with internal KKT
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns

    copyto!(kkt.inner.pr_diag, 1, kkt.pr_diag, shift_x + 1, nx)
    copyto!(kkt.inner.pr_diag, 1 + nx, kkt.pr_diag, shift_u + 1, nu)
    copyto!(kkt.inner.pr_diag, 1 + nx + nu, kkt.pr_diag, shift_s + 1, ns)

    # We dispatch the regularization evenly across the different processes
    kkt.inner.pr_diag[1+nx:nx+nu] ./= kkt.nblocks

    shift_c = id * m
    copyto!(kkt.inner.du_diag, 1, kkt.du_diag, shift_c + 1, m)
    return
end

function MadNLP.set_aug_diagonal!(kkt::SchurKKTSystem{T}, solver::MadNLP.MadNLPSolver) where T
    fill!(kkt.reg, zero(T))
    fill!(kkt.du_diag, zero(T))
    kkt.l_diag .= solver.xl_r .- solver.x_lr
    kkt.u_diag .= solver.x_ur .- solver.xu_r
    copyto!(kkt.l_lower, solver.zl_r)
    copyto!(kkt.u_lower, solver.zu_r)
    # Build pr_diag
    MadNLP._set_aug_diagonal!(kkt)
    # TODO : copy l,u diag to kkt.inner
    _synchronize_regularization!(kkt)
    return
end

function MadNLP.regularize_diagonal!(kkt::SchurKKTSystem, primal, dual)
    kkt.reg .+= primal
    kkt.pr_diag .+= primal
    kkt.du_diag .= .-dual
    _synchronize_regularization!(kkt)
    return
end
