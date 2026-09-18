# MPI cannot be initialized again after finalization. Each ownership scenario
# therefore starts a fresh Julia process, independent of this suite's MPI state.
@testset "MPI ownership in separate processes" begin
    worker = joinpath(@__DIR__, "mpi_ownership_worker.jl")
    project = dirname(Base.active_project())
    scenarios = (
        "caller-owned",
        "dmrg-owned-success",
        "dmrg-owned-failure",
        "dmrg-partial-init-failure",
        "dmrg-engine-cleanup-failure",
        "dmrg-calculation-and-cleanup-failure",
        "table3nu-owned-success",
        "table3nu-owned-failure",
        "table4-owned-success",
        "table4-owned-failure",
        "table-partial-init-failure",
        "table-cleanup-failure",
        "table-calculation-and-cleanup-failure",
    )
    for scenario in scenarios
        @testset "$scenario" begin
            mktempdir() do workdir
                logfile = joinpath(workdir, "worker.log")
                command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$project $worker $scenario`
                command = addenv(Cmd(command; dir = workdir), "OPENBLAS_NUM_THREADS" => "1")
                process = open(logfile, "w") do output
                    run(pipeline(command; stdout = output, stderr = output); wait = false)
                end
                # A hang must fail CI instead of leaving a persistent MPI job.
                status = timedwait(() -> process_exited(process), 180; pollint = 0.1)
                if status == :timed_out
                    kill(process, Base.SIGKILL)
                end
                wait(process)
                output = read(logfile, String)
                passed = status == :ok && success(process)
                if !passed
                    println(stderr, "MPI ownership scenario $scenario failed:\n", output)
                end
                @test status == :ok
                @test success(process)
                @test occursin("MPI_OWNERSHIP_OK $scenario", output)
            end
        end
    end
end
