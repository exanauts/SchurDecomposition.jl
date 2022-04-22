
#=

    Ordering: [x1, ..., xN, u]

    with:
    - xi: local variable for problem i-th
    - u : coupling variable
=#

struct BlockOPFModel <: NLPModels.AbstractNLPModel{Float64, Vector{Float64}}
    meta::NLPModels.NLPModelMeta{Float64, Vector{Float64}}
    counters::NLPModels.Counters
    nblocks::Int
    nx::Int
    nu::Int
    models::Vector{Argos.OPFModel}
    xs::Vector{Vector{Float64}}
    gs::Vector{Vector{Float64}}
    hash_x::Ref{UInt64}
    timers::Argos.NLPTimers
    etc::Dict{Symbol, Any}
end

function BlockOPFModel(nlps::Vector)
    models = [Argos.OPFModel(nlp) for nlp in nlps]

    nb = length(nlps)
    nx = nlps[1].nx
    nu = nlps[1].nu
    xl, xu = Argos.bounds(nlps[1], Argos.Variables())
    xlx, xux = xl[1:nx], xu[1:nx]
    xlu, xuu = xl[1+nx:nx+nu], xu[1+nx:nx+nu]
    bxl = [repeat(xlx, nb); xlu]
    bxu = [repeat(xux, nb); xuu]

    x00 = Argos.initial(nlps[1])
    x0 = [repeat(x00[1:nx], nb); x00[nx+1:nx+nu]]

    gl, gu = Argos.bounds(nlps[1], Argos.Constraints())

    xs = [zeros(Argos.n_variables(nlp)) for nlp in nlps]
    gs = [zeros(Argos.n_variables(nlp)) for nlp in nlps]
    etc = Dict{Symbol, Any}()

    ncon = sum(NLPModels.get_ncon.(models))
    y0 = zeros(ncon)

    return BlockOPFModel(
        NLPModels.NLPModelMeta(
            nx*nb+nu,
            ncon=ncon,
            nnzj = 0,
            nnzh = 0,
            x0 = x0,
            y0 = y0,
            lvar = bxl,
            uvar = bxu,
            lcon = repeat(gl, nb),
            ucon = repeat(gu, nb),
            minimize = true
        ),
        NLPModels.Counters(),
        length(nlps),
        nx, nu,
        models,
        xs,
        gs,
        Ref(UInt64(0)),
        Argos.NLPTimers(),
        Dict{Symbol, Any}(),
    )
end

function _update!(opf::BlockOPFModel, x::AbstractVector)
    hx = hash(x)
    if hx != opf.hash_x[]
        opf.timers.update_time += @elapsed begin
            shift_u = opf.nx*opf.nblocks
            for (idx, m) in enumerate(opf.models)
                xi = opf.xs[idx]
                shift_x = opf.nx * (idx - 1)
                copyto!(xi, 1, x, shift_x+1, opf.nx)
                copyto!(xi, 1+opf.nx, x, shift_u+1, opf.nu)
            end
        end
        opf.hash_x[] = hx
    end
end

# Objective
function NLPModels.obj(opf::BlockOPFModel, x::AbstractVector)
    _update!(opf, x)
    opf.timers.obj_time += @elapsed begin
        obj = 0.0
        for (idx, m) in enumerate(opf.models)
            obj += NLPModels.obj(m, opf.xs[idx])
        end
    end
    return obj
end

# Gradient
function NLPModels.grad!(opf::BlockOPFModel, x::AbstractVector, g::AbstractVector)
    fill!(g, 0.0)
    _update!(opf, x)
    shift_u = opf.nx * opf.nblocks
    gu = view(g, shift_u+1:shift_u+opf.nu)
    opf.timers.grad_time += @elapsed begin
        for (idx, m) in enumerate(opf.models)
            NLPModels.grad!(m, opf.xs[idx], opf.gs[idx])
        end
        for (idx, m) in enumerate(opf.models)
            copyto!(g, (idx-1)*opf.nx+1, opf.gs[idx], 1, opf.nx)
            gu .+= opf.gs[idx][opf.nx+1:opf.nx+opf.nu]
        end
    end
    return
end

# Constraints
function NLPModels.cons!(opf::BlockOPFModel, x::AbstractVector, c::AbstractVector)
    _update!(opf, x)
    opf.timers.cons_time += @elapsed begin
        shift = 0
        for (idx, m) in enumerate(opf.models)
            nc = NLPModels.get_ncon(m)
            ci = view(c, shift+1:shift+nc)
            NLPModels.cons!(m, opf.xs[idx], ci)
            shift += nc
        end
    end
    return
end

# Jacobian: sparse callback
function NLPModels.jac_coord!(m::BlockOPFModel, x::AbstractVector, jac::AbstractVector)
    # We should never assemble the full Jacobian
    @assert length(jac) == 0
    _update!(m, x)
    m.timers.jacobian_time += @elapsed begin
        for (idx, m) in enumerate(opf.models)
            # TODO
            NLPModels.jac_coord!(m.nlp, opf.xs[idx], 0)
        end
    end
    return
end

# Hessian: sparse callback
function NLPModels.hess_coord!(m::BlockOPFModel,x::AbstractVector, l::AbstractVector, hess::AbstractVector; obj_weight=1.0)
    # We should never assemble the full Hessian
    @assert length(hess) == 0
    m.timers.hessian_time += @elapsed begin
        shift = 0
        for (idx, m) in enumerate(opf.models)
            nc = NLPModels.get_ncon(m)
            yi = view(l, shift+1:shift+nc)
            NLPModels.hess_coord!(m.nlp, opf.xs[idx], yi, 0; obj_weight=obj_weight)
            shift += nc
        end
    end
    return
end

