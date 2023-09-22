#=
    Script to generate ma27/ma57 results for Figure 5.

    N.B.: ma57 is way slower than ma27, investigate why.
=#

include("common.jl")

using MadNLPHSL

function bench_sparse_kkt(model, nscen, linear_solver; ntrials=10)
    pload, qload = generate_loads(model, nscen, 0.0)
    nlp = Argos.StochEvaluator(model, pload, qload)
    blk = Argos.OPFModel(nlp)

    solver = MadNLP.MadNLPSolver(
        blk;
        linear_solver=linear_solver,
    )
    MadNLP.initialize!(solver)
    name = MadNLP.introduce(solver.linear_solver)
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
        name=name,
        time_build_kkt = t_build / ntrials,
        time_factorization = t_factorize / ntrials,
        time_backsolve = t_backsolve / ntrials,
    )
end

function run_benchmark_sparse_kkt(casename; nscens=[10, 20, 30, 60, 120, 240], ntrials=10, linear_solver=Ma27Solver)
    @info "Benchmark: " casename
    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)
    nexp = length(nscens)

    name = ""
    results = zeros(nexp, 4)
    for (i, k) in enumerate(nscens)
        println("    nscen: ", k)
        r = bench_sparse_kkt(model, k, linear_solver; ntrials=ntrials)
        results[i, :] .= (k, r.time_build_kkt, r.time_factorization, r.time_backsolve)
        name = r.name
    end

    if SAVE_RESULTS
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "$(casename)_sparse_kkt_$(name)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

