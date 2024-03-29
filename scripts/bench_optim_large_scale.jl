#=
    Script to generate results for Figure 8.
=#

include("common.jl")

function bench_optim(model, nscen; ntrials=3)
    blk = load_model(model, nscen, comm)
    instantiate_model!(blk)

    t_total, t_callbacks, t_linear_solver, t_reduce, t_comm = (0.0, 0.0, 0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    # Warm-up
    solver = build_solver(blk; max_iter=1)
    MadNLP.solve!(solver)
    for _ in 1:ntrials
        solver = build_solver(blk)
        MadNLP.solve!(solver)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
        t_reduce += solver.kkt.etc[:reduction]
        t_comm += solver.kkt.etc[:comm]
    end
    return (
        iters = n_it / ntrials,
        obj = obj / ntrials,
        total = t_total / ntrials,
        callbacks = t_callbacks / ntrials,
        linear_solver = t_linear_solver / ntrials,
        reduction = t_reduce / ntrials,
        comm = t_comm / ntrials,
    )
end

function run_benchmark_optim(casename; nscens=[10, 20, 30, 60, 120, 240])
    @info "Benchmark $casename"
    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)
    nexp = length(nscens)

    results = zeros(nexp, 8)
    for (i, k) in enumerate(nscens)
        println("    nscens=$(k)")
        r = bench_optim(model, k)
        results[i, :] .= (k, r.iters, r.obj, r.total, r.callbacks, r.linear_solver, r.reduction, r.comm)
        refresh_memory()
    end

    if SAVE_RESULTS
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "$(casename)_large_scale_$(nblk)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

run_benchmark_optim("case1354pegase"; nscens=[512])

