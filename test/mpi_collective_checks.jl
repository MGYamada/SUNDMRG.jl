module MPICollectiveChecks

using Test
using SUNDMRG

const MPI = SUNDMRG.MPI

@noinline _throw_probe(value) = throw(value)

function _capture(f)
    try
        f()
        return (raised = false, error = nothing, backtrace = nothing)
    catch error
        return (raised = true, error = error, backtrace = catch_backtrace())
    end
end

mutable struct EngineProbe
    failing_rank::Int
    attempts::Int
    finalizations::Int
    finalized_while_mpi_live::Bool
    failure::ArgumentError
end

abstract type PeerInitCPUEngine <: SUNDMRG.CPUEngine end
const active_engine_probe = Ref{Union{Nothing, EngineProbe}}(nothing)

function SUNDMRG._init_engine_runtime!(::Type{PeerInitCPUEngine}, rank, nranks)
    probe = active_engine_probe[]::EngineProbe
    probe.attempts += 1
    rank == probe.failing_rank && _throw_probe(probe.failure)
    return nothing
end

function SUNDMRG._finalize_engine_runtime!(::Type{PeerInitCPUEngine})
    probe = active_engine_probe[]::EngineProbe
    probe.finalizations += 1
    probe.finalized_while_mpi_live = MPI.Initialized() && !MPI.Finalized()
    return nothing
end

function run_checks(comm)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    expected_sum = nranks * (nranks + 1) ÷ 2

    @testset "Collective local errors, rank $rank of $nranks" begin
        @test SUNDMRG._collective_local(() -> (rank, :local), comm, rank, nranks, "local values") == (rank, :local)
        @test SUNDMRG._collective_local(() -> nothing, comm, rank, nranks, "nothing value") === nothing

        for failing_rank in 0 : nranks - 1
            failure = ArgumentError("local failure on rank $rank")
            outcome = _capture() do
                SUNDMRG._collective_local(comm, rank, nranks, "test local phase") do
                    rank == failing_rank && _throw_probe(failure)
                    :success
                end
            end
            @test outcome.raised
            @test occursin("local failure on rank $failing_rank", sprint(showerror, outcome.error))
            if rank == failing_rank
                @test outcome.error === failure
                @test any(frame -> frame.func == :_throw_probe, stacktrace(outcome.backtrace))
            else
                @test outcome.error isa ErrorException
                @test occursin("test local phase", sprint(showerror, outcome.error))
                @test occursin("rank $failing_rank", sprint(showerror, outcome.error))
            end
            @test MPI.Allreduce(rank + 1, +, comm) == expected_sum
        end

        failure = ArgumentError("simultaneous failure on rank $rank")
        outcome = _capture() do
            SUNDMRG._collective_local(() -> _throw_probe(failure), comm, rank, nranks, "simultaneous phase")
        end
        @test outcome.raised
        @test outcome.error === failure
        @test any(frame -> frame.func == :_throw_probe, stacktrace(outcome.backtrace))
        @test MPI.Allreduce(rank + 1, +, comm) == expected_sum

        # Julia permits any thrown value, including the successful callback's
        # usual return value. Its failure marker must remain unambiguous.
        failing_rank = nranks - 1
        outcome = _capture() do
            SUNDMRG._collective_local(comm, rank, nranks, "nothing exception") do
                rank == failing_rank && _throw_probe(nothing)
                :success
            end
        end
        @test outcome.raised
        if rank == failing_rank
            @test outcome.error === nothing
            @test any(frame -> frame.func == :_throw_probe, stacktrace(outcome.backtrace))
        else
            @test outcome.error isa ErrorException
            @test occursin("nothing exception", sprint(showerror, outcome.error))
            @test occursin("rank $failing_rank", sprint(showerror, outcome.error))
        end
        @test MPI.Allreduce(rank + 1, +, comm) == expected_sum
    end

    @testset "Peer engine initialization rollback, rank $rank of $nranks" begin
        previous_probe = active_engine_probe[]
        try
            for failing_rank in 0 : nranks - 1
                probe = EngineProbe(failing_rank, 0, 0, false, ArgumentError("engine failure on rank $failing_rank"))
                active_engine_probe[] = probe
                outcome = _capture() do
                    SUNDMRG._init_runtime_and_engine(PeerInitCPUEngine, :square, 2, 2, 2, rank, nranks)
                end
                @test outcome.raised
                @test probe.attempts == 1
                @test probe.finalizations == (rank == failing_rank ? 0 : 1)
                if rank == failing_rank
                    @test outcome.error === probe.failure
                    @test any(frame -> frame.func == :_throw_probe, stacktrace(outcome.backtrace))
                else
                    @test probe.finalized_while_mpi_live
                    @test outcome.error isa ErrorException
                    @test occursin("engine initialization", sprint(showerror, outcome.error))
                    @test occursin("engine failure on rank $failing_rank", sprint(showerror, outcome.error))
                end
                @test MPI.Allreduce(rank + 1, +, comm) == expected_sum
            end
        finally
            active_engine_probe[] = previous_probe
        end
    end

    return nothing
end

end # module MPICollectiveChecks
