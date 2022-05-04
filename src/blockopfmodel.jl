
#=

    Ordering: [x1, ..., xN, u]

    with:
    - xi: local variable for problem i-th
    - u : coupling variable
=#

struct BlockOPFModel <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    id::Int # warning: id is 0-based!
    nblocks::Int
    nx::Int
    nu::Int
    model::Argos.OPFModel
    x::Vector{Float64}
    xs::Vector{Float64}
    g::Vector{Float64}
    c::Vector{Float64}
    hash_x::Ref{UInt64}
    timers::Argos.NLPTimers
    comm::Union{Nothing, MPI.Comm}
    etc::Dict{Symbol, Any}
end

function BlockOPFModel(
    datafile::String,
    ploads::Array{Float64, 2},
    qloads::Array{Float64, 2},
    id::Int,
    nscen::Int,
    nblocks::Int;
    device=KA.CPU(),
    comm=nothing,
)
    @assert nscen % nblocks == 0
    Δ = div(nscen, nblocks)
    shift = id * Δ
    span = (shift+1):(shift+Δ)
    pl = ploads[:, span]
    ql = qloads[:, span]
    nlp = Argos.StochEvaluator(datafile, pl, ql; device=device)
    model = if isa(device, KA.CPU)
        Argos.OPFModel(nlp)
    else
        Argos.OPFModel(Argos.bridge(nlp))
    end

    nx = nlp.nx
    nu = nlp.nu
    n = Argos.n_variables(nlp)
    @assert n == nx + nu
    xl = NLPModels.get_lvar(model)
    xu = NLPModels.get_uvar(model)
    xlx, xux = xl[1:nx], xu[1:nx]
    xlu, xuu = xl[1+nx:nx+nu], xu[1+nx:nx+nu]
    bxl = [repeat(xlx, nblocks); xlu]
    bxu = [repeat(xux, nblocks); xuu]

    x00 = NLPModels.get_x0(model)
    x0 = [repeat(x00[1:nx], nblocks); x00[nx+1:nx+nu]]

    gl = NLPModels.get_lcon(model)
    gu = NLPModels.get_ucon(model)

    xs = zeros(n)
    x = zeros(nx*nblocks + nu)
    g = zeros(nx + nu)
    etc = Dict{Symbol, Any}()

    ncon = NLPModels.get_ncon(model) * nblocks
    y0 = zeros(ncon)
    c = zeros(ncon)

    return BlockOPFModel(
        NLPModels.NLPModelMeta(
            nx*nblocks+nu,
            ncon=ncon,
            nnzj = NLPModels.get_nnzj(model),
            nnzh = NLPModels.get_nnzh(model),
            x0 = x0,
            y0 = y0,
            lvar = bxl,
            uvar = bxu,
            lcon = repeat(gl, nblocks),
            ucon = repeat(gu, nblocks),
            minimize = true
        ),
        NLPModels.Counters(),
        id,
        nblocks,
        nx, nu,
        model,
        x,
        xs,
        g,
        c,
        Ref(UInt64(0)),
        Argos.NLPTimers(),
        comm,
        Dict{Symbol, Any}(),
    )
end

#=
    Update primal values in all MPI processes.

    We first update the primal values in the root process,
    then broadcast it to all the subprocesses.
    The local state xs is deduced from the id of the rank.
=#
function _update!(opf::BlockOPFModel, x::AbstractVector)
    hx = hash(x)
    if hx != opf.hash_x[]
        copyto!(opf.x, x)
        opf.hash_x[] = hx
        # Copy values internally in opf.xs
        shift_u = opf.nx*opf.nblocks
        copyto!(opf.xs, 1, opf.x, opf.id * opf.nx + 1, opf.nx)
        copyto!(opf.xs, opf.nx+1, opf.x, shift_u+1, opf.nu)
    end
end

# Objective
function NLPModels.obj(opf::BlockOPFModel, x::AbstractVector)
    _update!(opf, x)
    obj = NLPModels.obj(opf.model, opf.xs)
    # Accumulate objective along all subprocesses.
    cum_obj = comm_sum(obj, opf.comm)
    return cum_obj / opf.nblocks
end

# Gradient
function NLPModels.grad!(opf::BlockOPFModel, x::AbstractVector, g::AbstractVector)
    fill!(g, 0.0)
    _update!(opf, x)
    shift_u = opf.nx * opf.nblocks
    opf.timers.grad_time += @elapsed begin
        NLPModels.grad!(opf.model, opf.xs, opf.g)
        opf.g ./= opf.nblocks # scale down gradient
        # / State
        copyto!(g, opf.id*opf.nx+1, opf.g, 1, opf.nx)
        # / Coupling
        copyto!(g, shift_u+1, opf.g, opf.nx+1, opf.nu)
    end
    # Accumulate gradient on all subprocesses
    comm_sum!(g, opf.comm)
    return
end

# Constraints
function NLPModels.cons!(opf::BlockOPFModel, x::AbstractVector, c::AbstractVector)
    fill!(c, 0.0)
    _update!(opf, x)
    m = NLPModels.get_ncon(opf.model)
    shift = opf.id * m
    opf.timers.cons_time += @elapsed begin
        ci = view(c, shift+1:shift+m)
        NLPModels.cons!(opf.model, opf.xs, ci)
    end
    # Accumulate constraints on all processes
    comm_sum!(c, opf.comm)
    return
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(opf::BlockOPFModel, x::AbstractVector, jac::AbstractVector)
    _update!(opf, x)
    opf.timers.jacobian_time += @elapsed begin
        # Update internally Jacobian with AD
        NLPModels.jac_coord!(opf.model, opf.xs, jac)
    end
    return
end

# Hessian: sparse callback
function NLPModels.hess_coord!(opf::BlockOPFModel, x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight=1.0)
    _update!(opf, x)
    opf.timers.hessian_time += @elapsed begin
        m = NLPModels.get_ncon(opf.model)
        shift = opf.id * m
        yi = view(l, shift+1:shift+m)
        σ = obj_weight / opf.nblocks
        NLPModels.hess_coord!(opf.model, opf.xs, yi, hess; obj_weight=σ)
    end
    return
end

# Special scaling to keep Jacobian local
function MadNLP.scale_constraints!(
    opf::BlockOPFModel,
    con_scale::AbstractVector,
    jac::MadNLP.SparseMatrixCOO; # Ji
    max_gradient=1e-8,
)
    m = size(jac, 1) # number of local constraints
    shift_c = opf.id * m
    fill!(con_scale, 0.0)
    for i in 1:length(jac.I)
        row = @inbounds jac.I[i]
        @assert 1 <= row <= m
        @inbounds con_scale[row+shift_c] = max(con_scale[row+shift_c], abs(jac.V[i]))
    end

    comm_sum!(con_scale, opf.comm)
    con_scale .= min.(1.0, max_gradient ./ con_scale)
end

