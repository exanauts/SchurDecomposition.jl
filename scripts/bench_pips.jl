#=
    Script to generate PIPS results for Figure 5.
=#

include("common.jl")

function build_pips_solver(blk::SchurDecomposition.BlockOPFModel)
    madnlp_options = Dict{Symbol, Any}(
        :dual_initialized=>true,
        :lapack_algorithm=>MadNLP.BUNCHKAUFMAN,
        :linear_solver=>LAPACK_SOLVER,
        :max_iter=>250,
        :print_level=>MadNLP.ERROR,
        :tol=>1e-5,
    )
    KKT = SchurDecomposition.ParallelKKTSystem{Float64, Vector{Float64}, Matrix{Float64}}
    opt_ipm, opt_linear, logger = MadNLP.load_options(; madnlp_options...)
    return MadNLP.MadNLPSolver{Float64, KKT}(blk, opt_ipm, opt_linear; logger=logger)
end

function bench_pips_kkt(model, nscen; ntrials=10)
    blk = load_model(model, nscen, comm)
    solver = build_pips_solver(blk)
    MadNLP.initialize!(solver)

    t_build, t_factorize, t_backsolve = (0.0, 0.0, 0.0)
    # Warm-up
    MadNLP.build_kkt!(solver.kkt)
    MadNLP.factorize!(solver.linear_solver)
    MadNLP.solve_refine_wrapper!(solver, solver.d, solver.p)
    # Reset timers
    solver.kkt.etc[:comm] = 0.0
    solver.kkt.etc[:reduction] = 0.0
    # Benchmark
    for _ in 1:ntrials
        t_build     += @elapsed MadNLP.build_kkt!(solver.kkt)
        t_factorize += @elapsed MadNLP.factorize!(solver.linear_solver)
        t_backsolve += @elapsed MadNLP.solve_refine_wrapper!(solver,solver.d,solver.p)
    end
    return (
        time_build_kkt = t_build / ntrials,
        time_factorization = t_factorize / ntrials,
        time_backsolve = t_backsolve / ntrials,
        time_reduction = solver.kkt.etc[:reduction] / ntrials,
        time_comm = solver.kkt.etc[:comm] / ntrials,
    )
end

function run_benchmark_pips_kkt(casename, nscens; ntrials=10)
    datafile = joinpath(DATA, "$(casename).m")
    model = ExaPF.PolarForm(datafile, DEVICE)
    nexp = length(nscens)

    results = zeros(nexp, 6)
    for (i, k) in enumerate(nscens)
        r = bench_pips_kkt(model, k; ntrials=ntrials)
        results[i, :] .= (k, r.time_build_kkt, r.time_factorization, r.time_backsolve, r.time_reduction, r.time_comm)
    end

    if SAVE_RESULTS
        output_dir = joinpath(dirname(@__FILE__), RESULTS_DIR)
        if !isdir(output_dir)
            mkdir(output_dir)
        end
        output_file = joinpath(output_dir, "$(casename)_pips_kkt_$(nblk)_$(dev).txt")
        writedlm(output_file, results)
    end
    return results
end

run_benchmark_pips_kkt("case1354pegase", [2^i for i in 3:9]; ntrials=5)

