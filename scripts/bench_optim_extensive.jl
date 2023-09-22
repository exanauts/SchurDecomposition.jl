# Solve problem in extensive form with sparse linear algebra

include("common.jl")

using MadNLPHSL

function build_ext_solver(blk::Argos.OPFModel)
    return MadNLP.MadNLPSolver(
        blk;
        dual_initialized=true,
        linear_solver=Ma27Solver,
        max_iter=250,
        print_level=MadNLP.DEBUG,
        tol=1e-5,
    )
end

function gen_optim(model, nscen)
    pload, qload = generate_loads(model, nscen, 0.0)
    nlp = Argos.StochEvaluator(model, pload, qload)
    blk = Argos.OPFModel(nlp)

    # Instantiate
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    return build_ext_solver(blk)
end

function bench_optim_extensive(model, nscen; ntrials=3)
    pload, qload = generate_loads(model, nscen, 0.0)
    nlp = Argos.StochEvaluator(model, pload, qload)
    blk = Argos.OPFModel(nlp)

    # Instantiate
    x0 = NLPModels.get_x0(blk)
    nnzj = NLPModels.get_nnzj(blk)
    jac = zeros(nnzj)
    NLPModels.jac_coord!(blk, x0, jac)

    t_total, t_callbacks, t_linear_solver = (0.0, 0.0, 0.0)
    n_it = 0
    obj = 0

    solver = build_ext_solver(blk)
    # Warm-up
    MadNLP.solve!(solver)

    for _ in 1:ntrials
        solver = build_ext_solver(blk)
        MadNLP.solve!(solver)
        t_total += solver.cnt.total_time
        t_callbacks += solver.cnt.eval_function_time
        t_linear_solver += solver.cnt.linear_solver_time
        n_it += solver.cnt.k
        obj += solver.obj_val
    end
    return (
        iters = n_it / ntrials,
        obj = obj / ntrials,
        total = t_total / ntrials,
        callbacks = t_callbacks / ntrials,
        linear_solver = t_linear_solver / ntrials,
    )
end

function run_benchmark_optim_extensive(casename; nscens=[10, 20, 30, 60, 120, 240])
    @info "Benchmark $casename"
    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)
    nexp = length(nscens)

    results = zeros(nexp, 6)
    for (i, k) in enumerate(nscens)
        println("    nscens=$(k)")
        r = bench_optim_extensive(model, k)
        results[i, :] .= (k, r.iters, r.obj, r.total, r.callbacks, r.linear_solver)
        refresh_memory()
    end

    if SAVE_RESULTS
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "$(casename)_extoptim_$(nblk)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

