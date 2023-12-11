#=
    Script to generate results for Figure 4.
=#

include("common.jl")

function bench_callbacks(model, nscen; ntrials=10)
    blk = load_model(model, nscen, comm)

    n = NLPModels.get_nvar(blk)
    m = NLPModels.get_ncon(blk)
    x0 = NLPModels.get_x0(blk)

    obj = NLPModels.obj(blk, x0)
    tobj = @elapsed for i in 1:ntrials
        NLPModels.obj(blk, x0)
    end

    c = zeros(m)
    NLPModels.cons!(blk, x0, c)
    tcons = @elapsed for i in 1:ntrials
        NLPModels.cons!(blk, x0, c)
    end

    g = zeros(n)
    NLPModels.grad!(blk, x0, g)
    tgrad = @elapsed for i in 1:ntrials
        NLPModels.grad!(blk, x0, g)
    end

    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)
    tjac = @elapsed for i in 1:ntrials
        NLPModels.jac_coord!(blk, x0, jac)
    end

    nnzh = NLPModels.get_nnzh(blk)
    hess = zeros(nnzh)
    y0 = ones(m)
    NLPModels.hess_coord!(blk, x0, y0, hess)
    thess = @elapsed for i in 1:ntrials
        NLPModels.hess_coord!(blk, x0, y0, hess)
    end

    return (tobj, tcons, tgrad, tjac, thess)
end

function run_benchmark_callbacks(casename, nscens=[10, 20, 30, 60, 120, 240])
    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)
    nexp = length(nscens)

    results = zeros(nexp, 6)
    for (i, k) in enumerate(nscens)
        r = bench_callbacks(model, k)
        results[i, :] .= (k, r[1], r[2], r[3], r[4], r[5])
    end

    if SAVE_RESULTS
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "$(casename)_callbacks_$(nblk)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

run_benchmark_callbacks("case1354pegase", [2^i for i in 3:9])

