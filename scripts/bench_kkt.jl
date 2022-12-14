include("common.jl")

function bench_kkt(model, nscen; ntrials=10)
    blk = load_model(model, nscen, comm)
    instantiate_model!(blk)
    solver = build_solver(blk)

    t_build, t_factorize, t_backsolve = (0.0, 0.0, 0.0)
    # Warm-up
    MadNLP.build_kkt!(solver.kkt)
    MadNLP.factorize!(solver.linear_solver)
    MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)
    for _ in 1:ntrials
        t_build     += @elapsed MadNLP.build_kkt!(solver.kkt)
        t_factorize += @elapsed MadNLP.factorize!(solver.linear_solver)
        t_backsolve += @elapsed MadNLP.solve_refine_wrapper!(solver,solver.d,solver.p)
    end
    return (
        time_build_kkt = t_build / ntrials,
        time_factorization = t_factorize / ntrials,
        time_backsolve = t_backsolve / ntrials,
    )
end

function run_benchmark_kkt(casename; nscens=[10, 20, 30, 60, 120, 240])
    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)
    nexp = length(nscens)

    results = zeros(nexp, 4)
    for (i, k) in enumerate(nscens)
        r = bench_kkt(model, k)
        results[i, :] .= (k, r.time_build_kkt, r.time_factorization, r.time_backsolve)
    end

    if SAVE_RESULTS
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "$(casename)_kkt_$(nblk)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

run_benchmark_kkt("case1354pegase"; nscens=[2^i for i in 3:9])

