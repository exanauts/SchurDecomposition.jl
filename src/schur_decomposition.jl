
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

struct SchurKKTSystem{T, VI, VT, MT} <: MadNLP.AbstractReducedKKTSystem{T, MT}
    inner::Argos.BieglerKKTSystem{T, VI, VT, MT}
    id::Int
    nblocks::Int
    _w1::VT
    _w2::VT
    _w3::VT
    _w4::VT
    _w5::VT
    comm::Union{MPI.Comm, Nothing}
end

function SchurKKTSystem{T, VI, VT, MT}(
    blk::BlockOPFModel,
    ind_cons=MadNLP.get_index_constraints(blk);
    max_batches=256,
) where {T, VI, VT, MT}

    id = blk.id
    nblocks = blk.nblocks
    comm = blk.comm

    nlp = blk.model
    kkt = Argos.BieglerKKTSystem{T, VI, VT, MT}(nlp; max_batches=max_batches)

    n = NLPModels.get_nvar(nlp)
    m = NLPModels.get_ncon(nlp)
    ns = length(kkt.ind_ineq)
    # Buffers
    _w1 = zeros(n)
    _w2 = zeros(m)
    _w3 = zeros(n+ns)
    _w4 = zeros(n+ns+m)
    _w5 = zeros(n+ns+m)
    return SchurKKTSystem{T, VI, VT, MT}(kkt, id, nblocks, _w1, _w2, _w3, _w4, _w5, comm)
end

MadNLP.num_variables(kkt::SchurKKTSystem) = kkt.inner.nu
MadNLP.get_hessian(kkt::SchurKKTSystem) = MadNLP.get_hessian(kkt.inner)
MadNLP.get_jacobian(kkt::SchurKKTSystem) = MadNLP.get_jacobian(kkt.inner)
MadNLP.is_reduced(::SchurKKTSystem) = true
MadNLP.nnz_jacobian(kkt::SchurKKTSystem) = MadNLP.nnz_jacobian(kkt.inner)
MadNLP.get_kkt(kkt::SchurKKTSystem) = kkt.inner.aug_com


function Base.getproperty(blk::SchurKKTSystem, d::Symbol)
    if d === :inner
        return Base.getfield(blk, :inner)
    elseif d === :pr_diag
        return Base.getfield(blk.inner, :pr_diag)
    elseif d === :du_diag
        return Base.getfield(blk.inner, :du_diag)
    elseif hasproperty(blk, d)
        return Base.getfield(blk, d)
    else
        error("$d")
    end
end

MadNLP.initialize!(kkt::SchurKKTSystem) = MadNLP.initialize!(kkt.inner)
MadNLP.get_raw_jacobian(kkt::SchurKKTSystem) = MadNLP.get_raw_jacobian(kkt.inner)

function MadNLP.set_jacobian_scaling!(kkt::SchurKKTSystem{T,VI,VT,MT}, constraint_scaling::AbstractVector) where {T,VI,VT,MT}
    m = size(kkt.inner.J, 1)
    shift_c = kkt.id * m
    _c = view(constraint_scaling, shift_c+1:shift_c+m)
    MadNLP.set_jacobian_scaling!(kkt.inner, _c)
end

function MadNLP.mul!(y::AbstractVector, kkt::SchurKKTSystem, x::AbstractVector)
    mul!(y, kkt.inner, x)
end

function MadNLP.jtprod!(
    y_h::AbstractVector,
    kkt::SchurKKTSystem{T, VI, VT, MT},
    x_h::AbstractVector,
) where {T, VI, VT, MT}
    nx = kkt.inner.nx
    nu = kkt.inner.nu
    m = size(kkt.inner.J, 1)
    ns = length(kkt.inner.ind_ineq)
    shift_x = kkt.id * nx
    shift_u = kkt.nblocks * nx
    shift_s = kkt.nblocks * nx + nu + kkt.id * m
    shift_c = kkt.id * m

    _x = kkt._w2
    _y = kkt._w3

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

MadNLP.compress_jacobian!(kkt::SchurKKTSystem) = MadNLP.compress_jacobian!(kkt.inner)
MadNLP.compress_hessian!(kkt::SchurKKTSystem) = MadNLP.compress_hessian!(kkt.inner)

function MadNLP.build_kkt!(kkt::SchurKKTSystem)
    MadNLP.build_kkt!(kkt.inner)
    # Assemble Schur complement (reduction) on all processes
    comm_sum!(kkt.inner.aug_com, kkt.comm)
end

function MadNLP.solve_refine_wrapper!(
    ips::MadNLP.InteriorPointSolver{<:SchurKKTSystem{T,VI,VT,MT}},
    x_h, b_h,
) where {T, VI, VT, MT}
    comm = ips.kkt.comm
    kkt = ips.kkt.inner
    # MPI
    id = ips.kkt.id
    nblocks = ips.kkt.nblocks
    # Problem's dimension
    m = div(ips.m, nblocks)  # constraints
    nx, nu = kkt.nx, kkt.nu # state and control
    ns = length(kkt.ind_ineq)

    # Transfer
    shift_x = id * nx
    shift_u = nblocks * nx
    shift_s = nblocks * nx + nu + id * ns
    shift_y = nblocks * (nx+ns) + nu + id * m

    _x = ips.kkt._w4
    _b = ips.kkt._w5

    copyto!(_b,          1, b_h, shift_x + 1, nx)
    copyto!(_b,       nx+1, b_h, shift_u + 1, nu)
    copyto!(_b,    nx+nu+1, b_h, shift_s + 1, ns)
    copyto!(_b, nx+nu+ns+1, b_h, shift_y + 1, m)

    # Transfer data to device
    x = Argos._load_buffer(kkt, _x, :kkt_x)::VT
    b = Argos._load_buffer(kkt, _b, :kkt_b)::VT

    MadNLP.fixed_variable_treatment_vec!(b, ips.ind_fixed)
    @assert length(b) == length(x) == ips.n + m

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

    Λ = 1.0 ./ (Σd .- α.^2 ./ Σₛ)

    # RHS
    r₁₂ = view(b, 1:nx+nu)
    r₁ = view(b, 1:nx)                   # / state
    r₂ = view(b, 1+nx:nx+nu)             # / control
    r₃ = view(b, 1+nx+nu:ips.n)          # / slack
    r₄ = view(b, ips.n+1:ips.n+nx)       # / equality cons
    r₅ = view(b, ips.n+nx+1:ips.n+m)     # / inequality cons
    # LHS
    dxu = view(x, 1:nx+nu)
    dx = view(x, 1:nx)                   # / state
    du = view(x, 1+nx:nx+nu)             # / control
    ds = view(x, 1+nx+nu:ips.n)          # / slack
    dλ = view(x, ips.n+1:ips.n+nx)       # / equality cons
    dy = view(x, ips.n+nx+1:ips.n+m)     # / inequality cons

    # Reduction (1) --- Condensed
    vj .= Λ .* (r₅ .+ α .* r₃ ./ Σₛ)      # v = (α Σₛ⁻¹ α)⁻¹ * (r₅ + α Σₛ⁻¹ r₃)
    mul!(jv, kkt.A', vj, -1.0, 0.0)       # jᵥ = Aᵀ v
    jv .+= r₁₂                            # r₁₂ - Aᵀv
    # Reduction (2) --- Biegler
    sx1 .= r₄                             # r₄
    ldiv!(Gxi, sx1)                       # Gₓ⁻¹ r₄
    sx2 .= tx                             # tx = jv[1:nx]
    kvx .= sx1 ; kvu .= 0.0
    mul!(kh, K, kv)                       # [Kₓₓ Gₓ⁻¹ r₄ ; Kᵤₓ Gₓ⁻¹ r₄ ]
    sx2 .= khx .- tx                      # sₓ = Kₓₓ Gₓ⁻¹ r₄ - tₓ
    ldiv!(Gxi', sx2)                      # Gₓ⁻ᵀ sₓ
    mul!(tu, Gu', sx2, 1.0, 1.0)          # tᵤ = tᵤ + Gᵤᵀ Gₓ⁻ᵀ sₓ
    axpy!(-1.0, khu, tu)                  # tᵤ = tᵤ - Kᵤₓ Gₓ⁻¹ r₄

    du .= tu
    comm_sum!(du, comm)
    ips.cnt.linear_solver_time += @elapsed begin
        MadNLP.solve!(ips.linear_solver, du)
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

    MadNLP.fixed_variable_treatment_vec!(x, ips.ind_fixed)

    copyto!(x_h, shift_x + 1, x,          1, nx)
    copyto!(x_h, shift_u + 1, x,       nx+1, nu)
    copyto!(x_h, shift_s + 1, x,    nx+nu+1, ns)
    copyto!(x_h, shift_y + 1, x, nx+nu+ns+1, m)

    comm_sum!(x_h, comm)
    return solve_status
end

