
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

struct SchurKKTSystem{T,VI,VT,MT} <:
       MadNLP.AbstractReducedKKTSystem{T,VT,MT,MadNLP.ExactHessian{T,VT}}
    inner::Argos.BieglerKKTSystem{T,VI,VT,MT}
    id::Int
    nblocks::Int
    pr_diag::VT
    du_diag::VT
    _w1::VT
    _w2::VT
    _w3::VT
    _w4::VT
    _w5::VT
    comm::Union{MPI.Comm,Nothing}
    etc::Dict{Symbol,Any}
end

function SchurKKTSystem{T,VI,VT,MT}(
    blk::BlockOPFModel,
    ind_cons=MadNLP.get_index_constraints(blk);
    max_batches=256,
) where {T,VI,VT,MT}
    id = blk.id
    nblocks = blk.nblocks
    comm = blk.comm

    nlp = blk.model
    kkt = Argos.BieglerKKTSystem{T,VI,VT,MT}(nlp; max_batches=max_batches)
    nx = kkt.nx
    nu = kkt.nu

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    ns = length(kkt.ind_ineq)

    pr_diag = VT(undef, nblocks * (nx + ns) + nu)
    fill!(pr_diag, 0)
    du_diag = VT(undef, nblocks * m)
    fill!(du_diag, 0)

    # Buffers
    _w1 = zeros(n)
    _w2 = zeros(m)
    _w3 = zeros(n + ns)
    _w4 = zeros(n + ns + m)
    _w5 = zeros(n + ns + m)

    # Timers
    etc = Dict{Symbol,Any}()
    etc[:reduction] = 0.0
    etc[:comm] = 0.0
    etc[:backsolve] = 0.0

    return SchurKKTSystem{T,VI,VT,MT}(
        kkt,
        id,
        nblocks,
        pr_diag,
        du_diag,
        _w1,
        _w2,
        _w3,
        _w4,
        _w5,
        comm,
        etc,
    )
end

Base.size(kkt::SchurKKTSystem, n::Int) = size(kkt.inner, n)
Base.size(kkt::SchurKKTSystem) = size(kkt.inner)
MadNLP.num_variables(kkt::SchurKKTSystem) = kkt.inner.nu
MadNLP.get_hessian(kkt::SchurKKTSystem) = MadNLP.get_hessian(kkt.inner)
MadNLP.get_jacobian(kkt::SchurKKTSystem) = MadNLP.get_jacobian(kkt.inner)
MadNLP.is_reduced(::SchurKKTSystem) = true
MadNLP.nnz_jacobian(kkt::SchurKKTSystem) = MadNLP.nnz_jacobian(kkt.inner)
MadNLP.get_kkt(kkt::SchurKKTSystem) = kkt.inner.aug_com

MadNLP.initialize!(kkt::SchurKKTSystem) = MadNLP.initialize!(kkt.inner)
MadNLP.get_raw_jacobian(kkt::SchurKKTSystem) = MadNLP.get_raw_jacobian(kkt.inner)

function MadNLP.set_jacobian_scaling!(
    kkt::SchurKKTSystem{T,VI,VT,MT},
    constraint_scaling::AbstractVector,
) where {T,VI,VT,MT}
    m = size(kkt.inner.J, 1)
    shift_c = kkt.id * m
    _c = view(constraint_scaling, shift_c+1:shift_c+m)
    return MadNLP.set_jacobian_scaling!(kkt.inner, _c)
end

function MadNLP.mul!(y::AbstractVector, kkt::SchurKKTSystem, x::AbstractVector)
    return mul!(y, kkt.inner, x)
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

function MadNLP.solve_refine_wrapper!(
    solver::MadNLP.MadNLPSolver{T,<:SchurKKTSystem{T,VI,VT,MT}},
    x_r::MadNLP.AbstractKKTVector,
    b_r::MadNLP.AbstractKKTVector,
) where {T,VI,VT,MT}
    comm = solver.kkt.comm
    kkt = solver.kkt.inner
    # MPI
    id = solver.kkt.id
    nblocks = solver.kkt.nblocks
    # Problem's dimension
    m = div(solver.m, nblocks)  # constraints
    nx, nu = kkt.nx, kkt.nu # state and control
    ns = length(kkt.ind_ineq)
    n = nx + nu + ns # local number of variables, replace solver.n

    x_h = MadNLP.full(x_r)
    b_h = MadNLP.full(b_r)
    # Transfer
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns
    shift_y = nblocks * (nx + ns) + nu + id * m

    _x = solver.kkt._w4
    _b = solver.kkt._w5

    copyto!(_b, 1, b_h, shift_x + 1, nx)
    copyto!(_b, nx + 1, b_h, shift_u + 1, nu)
    copyto!(_b, nx + nu + 1, b_h, shift_s + 1, ns)
    copyto!(_b, nx + nu + ns + 1, b_h, shift_y + 1, m)

    # Transfer data to device
    x = Argos._load_buffer(kkt, _x, :kkt_x)::VT
    b = Argos._load_buffer(kkt, _b, :kkt_b)::VT

    MadNLP.fixed_variable_treatment_vec!(b, solver.ind_fixed)

    # Buffers
    jv = kkt._wxu1
    tx = view(jv, 1:nx)
    tu = view(jv, nx+1:nx+nu)
    vj = kkt._wj1
    sx1 = kkt._wx1
    sx2 = kkt._wx2
    # Views for Hessian-vector multiplication
    kv = kkt._wxu2
    kh = kkt._wxu3
    kvx = view(kv, 1:nx)
    kvu = view(kv, nx+1:nx+nu)
    khx = view(kh, 1:nx)
    khu = view(kh, nx+1:nx+nu)

    # Gₓ⁻¹
    Gxi = kkt.G_fac
    Gx = kkt.Gx
    Gu = kkt.Gu
    K = kkt.K
    Σₛ = view(kkt.pr_diag, nx+nu+1:nx+nu+ns)
    Σd = view(kkt.du_diag, nx+1:m)
    α = view(kkt.con_scale, nx+1:m)

    Λ = 1.0 ./ (Σd .- α .^ 2 ./ Σₛ)

    # RHS
    r₁₂ = view(b, 1:nx+nu)
    r₁ = view(b, 1:nx)                   # / state
    r₂ = view(b, 1+nx:nx+nu)             # / control
    r₃ = view(b, 1+nx+nu:n)              # / slack
    r₄ = view(b, n+1:n+nx)               # / equality cons
    r₅ = view(b, n+nx+1:n+m)             # / inequality cons
    # LHS
    dxu = view(x, 1:nx+nu)
    dx = view(x, 1:nx)                   # / state
    du = view(x, 1+nx:nx+nu)             # / control
    ds = view(x, 1+nx+nu:n)              # / slack
    dλ = view(x, n+1:n+nx)               # / equality cons
    dy = view(x, n+nx+1:n+m)             # / inequality cons

    r₂ ./= nblocks
    # Reduction (1) --- Condensed
    vj .= Λ .* (r₅ .+ α .* r₃ ./ Σₛ)      # v = (α Σₛ⁻¹ α)⁻¹ * (r₅ + α Σₛ⁻¹ r₃)
    mul!(jv, kkt.A', vj, -1.0, 0.0)       # jᵥ = Aᵀ v
    jv .+= r₁₂                            # r₁₂ - Aᵀv
    # Reduction (2) --- Biegler
    sx1 .= r₄                             # r₄
    ldiv!(Gxi, sx1)                       # Gₓ⁻¹ r₄
    sx2 .= tx                             # tx = jv[1:nx]
    kvx .= sx1
    kvu .= 0.0
    mul!(kh, K, kv)                       # [Kₓₓ Gₓ⁻¹ r₄ ; Kᵤₓ Gₓ⁻¹ r₄ ]
    sx2 .= khx .- tx                      # sₓ = Kₓₓ Gₓ⁻¹ r₄ - tₓ
    ldiv!(Gxi', sx2)                      # Gₓ⁻ᵀ sₓ
    mul!(tu, Gu', sx2, 1.0, 1.0)          # tᵤ = tᵤ + Gᵤᵀ Gₓ⁻ᵀ sₓ
    axpy!(-1.0, khu, tu)                  # tᵤ = tᵤ - Kᵤₓ Gₓ⁻¹ r₄

    du .= tu
    tic = comm_walltime(comm)
    comm_sum!(du, comm)
    solver.kkt.etc[:comm] += comm_walltime(comm) - tic
    solver.cnt.linear_solver_time += @elapsed begin
        MadNLP.solve!(solver.linear_solver, du)
    end
    solve_status = true

    # (1) Extract Biegler
    dx .= r₄                              # r₄
    mul!(dx, Gu, du, -1.0, 1.0)           # r₄ - Gᵤ dᵤ
    ldiv!(Gxi, dx)                        # dₓ = Gₓ⁻¹ (r₄ - Gᵤ dᵤ)
    dλ .= tx                              # tₓ
    mul!(kh, K, dxu)                      # Kₓₓ dₓ + Kₓᵤ dᵤ
    axpy!(-1.0, khx, dλ)                  # tₓ - Kₓₓ dₓ + Kₓᵤ dᵤ
    ldiv!(Gxi', dλ)                       # dₗ = Gₓ⁻ᵀ(tₓ - Kₓₓ dₓ + Kₓᵤ dᵤ)
    # (2) Extract Condensed
    mul!(vj, kkt.A, dxu)                  # Aₓ dₓ + Aᵤ dᵤ
    dy .= Λ .* (r₅ .- vj .+ α .* r₃ ./ Σₛ)
    ds .= (r₃ .+ α .* dy) ./ Σₛ

    MadNLP.fixed_variable_treatment_vec!(x, solver.ind_fixed)

    fill!(x_h, 0)
    copyto!(x_h, shift_x + 1, x, 1, nx)
    copyto!(x_h, shift_u + 1, x, nx + 1, nu)
    copyto!(x_h, shift_s + 1, x, nx + nu + 1, ns)
    copyto!(x_h, shift_y + 1, x, nx + nu + ns + 1, m)

    # Scale coupling's direction by number of processes
    x_h[shift_u+1:shift_u+nu] ./= nblocks

    tic = comm_walltime(comm)
    comm_sum!(x_h, comm)
    solver.kkt.etc[:comm] += comm_walltime(comm) - tic

    return solve_status
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

    # We dispatch the regularization evently across the different processes
    kkt.inner.pr_diag[1+nx:nx+nu] ./= kkt.nblocks

    shift_c = id * m
    return copyto!(kkt.inner.du_diag, 1, kkt.du_diag, shift_c + 1, m)
end

function MadNLP.set_aug_diagonal!(kkt::SchurKKTSystem, solver::MadNLP.MadNLPSolver)
    # Load data
    x = MadNLP.full(solver.x)
    xl = MadNLP.full(solver.xl)
    xu = MadNLP.full(solver.xu)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)
    _pr_diag = zeros(length(kkt.pr_diag))
    # Global regularization
    _pr_diag .= zl ./ (x .- xl) .+ zu ./ (xu .- x)
    copyto!(kkt.pr_diag, _pr_diag)
    fill!(kkt.du_diag, 0.0)
    return _synchronize_regularization!(kkt)
end

function MadNLP.regularize_diagonal!(kkt::SchurKKTSystem, primal, dual)
    kkt.pr_diag .+= primal
    kkt.du_diag .= .-dual
    return _synchronize_regularization!(kkt)
end
