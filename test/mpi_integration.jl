# Standalone entry point: julia --project=. --startup-file=no test/mpi_integration.jl
# Keep this parent outside MPI. Each launcher job has a separate MPI lifetime.
using Test
using MPI
using JLD2
using TOML

function stop_mpi_job(process; grace = 5)
    process_exited(process) && return nothing
    # Let the MPI launcher forward termination to ranks, including any that it
    # placed in their own process groups, before using the forced fallback.
    kill(process, Base.SIGTERM)
    timedwait(() -> process_exited(process), grace; pollint = 0.1) == :ok && return nothing
    group_signaled = Sys.isunix() &&
        ccall(:kill, Cint, (Cint, Cint), -getpid(process), Base.SIGKILL) == 0
    if !group_signaled && !process_exited(process)
        kill(process, Base.SIGKILL)
    end
    stopped = timedwait(() -> process_exited(process), grace; pollint = 0.1)
    stopped == :ok || error("MPI launcher could not be stopped after its timeout")
    return nothing
end

function launch_mpi_worker(worker, ranks, arguments, workdir; marker, timeout = 240)
    project = dirname(Base.active_project())
    command = `$(MPI.mpiexec()) -n $ranks $(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project $worker $arguments`
    # A separate process group lets timeout cleanup include the local MPI ranks.
    command = addenv(Cmd(command; dir = workdir, detach = Sys.isunix()), "OPENBLAS_NUM_THREADS" => "1")
    logfile = joinpath(workdir, "worker.log")
    println("Running MPI integration: $(basename(worker)), ranks=$ranks, arguments=$(join(arguments, ' '))")
    process = open(logfile, "w") do output
        run(pipeline(command; stdout = output, stderr = output); wait = false)
    end
    status = timedwait(() -> process_exited(process), timeout; pollint = 0.1)
    if status == :timed_out
        stop_mpi_job(process)
    end
    wait(process)
    output = read(logfile, String)
    passed = status == :ok && success(process) && occursin(marker, output)
    if !passed
        println(stderr, "MPI integration worker failed:\n", output)
    end
    @test status == :ok
    @test success(process)
    @test occursin(marker, output)
    return passed
end

function compare_mpi_results(reference, actual)
    @test Set(keys(actual)) == Set(keys(reference))
    for key in keys(reference)
        @testset "$(key)" begin
            expected = reference[key]
            result = actual[key]
            @test result.energy ≈ expected.energy atol = 1e-10 rtol = 0
            @test Set(keys(result.correlations)) == Set(keys(expected.correlations))
            if key[3] == 0
                # The ground singlet is unique. Excited states are degenerate,
                # so their entropy/correlations need not select the same basis.
                @test result.entropy ≈ expected.entropy atol = 1e-10 rtol = 0
                for bond in keys(expected.correlations)
                    @test result.correlations[bond] ≈ expected.correlations[bond] atol = 1e-10 rtol = 0
                end
            end
        end
    end
end

@testset "Standalone CPU MPI integration" begin
    @test !MPI.Initialized()
    mktempdir() do workspace
        results_by_size = Dict{Int, Any}()
        growth_by_size = Dict{Int, Any}()
        for ranks in (1, 2)
            @testset "$ranks ranks" begin
                workdir = joinpath(workspace, "success-$ranks")
                mkpath(workdir)
                passed = launch_mpi_worker(
                    joinpath(@__DIR__, "mpi_integration_worker.jl"), ranks,
                    [workdir, string(ranks)], workdir;
                    marker = "MPI_INTEGRATION_OK ranks=$ranks",
                )
                if passed
                    for rank in 0 : ranks - 1
                        @test read(joinpath(workdir, "rank-$rank.ok"), String) == "rank=$rank size=$ranks finalized=true\n"
                    end
                    results = load(joinpath(workdir, "results.jld2"), "results")
                    @test length(results) == 8
                    results_by_size[ranks] = results
                    growth_by_size[ranks] = load(joinpath(workdir, "results.jld2"), "growth_result")
                    # Compare storage backends and both Lanczos modes within
                    # each process count, in addition to comparing MPI sizes.
                    for target in (0, 1), alg in (:slow, :fast), fileio in (false, true)
                        alg == :slow && !fileio && continue
                        baseline = results[(:slow, false, target)]
                        compare_mpi_results(Dict((alg, fileio, target) => baseline),
                            Dict((alg, fileio, target) => results[(alg, fileio, target)]))
                    end
                end
            end
        end
        if length(results_by_size) == 2
            @testset "one-rank versus two-rank" begin
                compare_mpi_results(results_by_size[1], results_by_size[2])
                @test growth_by_size[1].energy ≈ growth_by_size[2].energy atol = 1e-10 rtol = 0
                @test growth_by_size[1].entropy ≈ growth_by_size[2].entropy atol = 1e-10 rtol = 0
            end
        end
        for scenario in (
            "caller_initial_output", "caller_sweep_output", "caller_invalid_input",
            "caller_invalid_rank0", "caller_cleanup", "owned_initial_output", "owned_sweep_output",
        )
            @testset "$scenario" begin
                workdir = joinpath(workspace, scenario)
                mkpath(workdir)
                passed = launch_mpi_worker(
                    joinpath(@__DIR__, "mpi_failure_scenarios.jl"), 2,
                    [scenario, workdir], workdir;
                    marker = "MPI_FAILURE_OK $scenario",
                )
                if passed
                    for rank in 0 : 1
                        report = TOML.parsefile(joinpath(workdir, "failure-$scenario-rank$rank.toml"))
                        @test report["scenario"] == scenario
                        @test report["rank"] == rank
                        @test report["nranks"] == 2
                        @test report["mpi_finalized"]
                        @test report["caller_reused"] == startswith(scenario, "caller_")
                        invalid = occursin("invalid", scenario)
                        @test report["initialized"] == (invalid ? 0 : 1)
                        @test report["finalized"] == (invalid ? 0 : 1)
                        @test report["cleaned"] == (invalid || rank != 0 ? 0 : 1)
                        @test !isempty(report["message"])
                    end
                end
            end
        end
    end
    @test !MPI.Initialized()
    @test !MPI.Finalized()
end
