module RuntimeFailureTests

using Test
using Random
using Logging
using MPI
using SUNDMRG

# These fixtures extend the existing engine and storage dispatch points. All
# numerical operations and JLD2 reads/writes use the actual CPU implementation.
Base.@kwdef mutable struct FailureProbe
    stage::Symbol = :none
    cleanup_fails::Bool = false
    finalize_fails::Bool = false
    initialized::Int = 0
    finalized::Int = 0
    cleaned::Int = 0
    disk::Union{Nothing, SUNDMRG.JLD2InternalStorage} = nothing
    saved_blocks::Set{Tuple{Symbol, Int}} = Set{Tuple{Symbol, Int}}()
    failure_operation::Symbol = :none
    failure_files::Vector{String} = String[]
    logs::Vector{Any} = Any[]
    calculation_error::ErrorException = ErrorException("injected calculation failure")
    cleanup_error::ErrorException = ErrorException("injected storage cleanup failure")
    finalize_error::ErrorException = ErrorException("injected engine finalization failure")
end

struct FailureScratch
    path::String
    probe::FailureProbe
end

struct FailureStorage <: SUNDMRG.AbstractInternalStorage
    disk::SUNDMRG.JLD2InternalStorage
    probe::FailureProbe
end

abstract type TrackingCPUEngine <: SUNDMRG.CPUEngine end
const active_probe = Ref{Union{Nothing, FailureProbe}}(nothing)

function SUNDMRG._init_engine_runtime!(::Type{TrackingCPUEngine}, rank, Ncpu)
    probe = active_probe[]::FailureProbe
    probe.initialized += 1
    return nothing
end

function SUNDMRG._finalize_engine_runtime!(::Type{TrackingCPUEngine})
    probe = active_probe[]::FailureProbe
    probe.finalized += 1
    probe.finalize_fails && throw(probe.finalize_error)
    return nothing
end

function fail_calculation!(probe, operation)
    probe.failure_operation = operation
    probe.failure_files = sort(readdir(SUNDMRG._storage_dir(probe.disk)))
    throw(probe.calculation_error)
end

function SUNDMRG.to_engine_array(::Type{TrackingCPUEngine}, array)
    probe = active_probe[]::FailureProbe
    if probe.stage == :after_storage
        fail_calculation!(probe, :after_storage)
    end
    return array
end

function SUNDMRG.init_internal_storage(fileio::Val{true}, scratch::FailureScratch,
        block_table, trmat_table, tensor_table, rank)
    disk = SUNDMRG.init_internal_storage(fileio, scratch.path,
        block_table, trmat_table, tensor_table, rank)
    scratch.probe.disk = disk
    return FailureStorage(disk, scratch.probe)
end

function SUNDMRG.save_block(storage::FailureStorage, label, len, block)
    probe = storage.probe
    # On the 2x2 fixture, warmup writes each block length once and no growth is
    # needed. The first rewrite is the right length-1 block in the first sweep.
    if probe.stage == :sweep && (label, len) in probe.saved_blocks
        fail_calculation!(probe, :sweep)
    end
    SUNDMRG.save_block(storage.disk, label, len, block)
    push!(probe.saved_blocks, (label, len))
    return nothing
end

function SUNDMRG.save_trmat(storage::FailureStorage, label, len, trmat)
    if storage.probe.stage == :initial_output && label == :l && len == 0
        fail_calculation!(storage.probe, :initial_output)
    end
    return SUNDMRG.save_trmat(storage.disk, label, len, trmat)
end

SUNDMRG.load_block(storage::FailureStorage, label, len) =
    SUNDMRG.load_block(storage.disk, label, len)
SUNDMRG.load_trmat(storage::FailureStorage, label, len) =
    SUNDMRG.load_trmat(storage.disk, label, len)
SUNDMRG.load_tensor(storage::FailureStorage, label, len, y) =
    SUNDMRG.load_tensor(storage.disk, label, len, y)
SUNDMRG.has_tensor(storage::FailureStorage, label, len, y) =
    SUNDMRG.has_tensor(storage.disk, label, len, y)
SUNDMRG.take_tensor!(storage::FailureStorage, label, len, y) =
    SUNDMRG.take_tensor!(storage.disk, label, len, y)
SUNDMRG.save_tensor(storage::FailureStorage, label, len, y, tensor) =
    SUNDMRG.save_tensor(storage.disk, label, len, y, tensor)

function SUNDMRG.cleanup_storage!(storage::FailureStorage)
    storage.probe.cleaned += 1
    storage.probe.cleanup_fails && throw(storage.probe.cleanup_error)
    return SUNDMRG.cleanup_storage!(storage.disk)
end

function run_probe(probe, scratch)
    previous_probe = active_probe[]
    active_probe[] = probe
    logger = Test.TestLogger()
    try
        return with_logger(logger) do
            Random.seed!(1858)
            run_DMRG(SU(2) * HeisenbergModel(), SquareLattice(2, 2),
                8, [8], 8, TrackingCPUEngine;
                fileio = true, scratch = FailureScratch(scratch, probe),
                max_cooldown_sweeps = 8, verbose = false, manage_mpi = false)
        end
    catch err
        return err
    finally
        append!(probe.logs, logger.logs)
        active_probe[] = previous_probe
    end
end

function test_reported_cleanup_errors(probe, expected)
    logs = filter(log -> log.level >= Logging.Error, probe.logs)
    @test length(logs) == length(expected)
    for (log, expected_error) in zip(logs, expected)
        @test haskey(log.kwargs, :exception)
        logged_error, logged_backtrace = log.kwargs[:exception]
        @test logged_error === expected_error
        @test !isempty(logged_backtrace)
        @test occursin(expected_error.msg, sprint(showerror, logged_error))
    end
end

function error_leaves(err)
    if err isa CompositeException
        return reduce(vcat, error_leaves.(err.exceptions); init = Any[])
    elseif err isa CapturedException
        return error_leaves(err.ex)
    end
    return Any[err]
end

function with_scratch(f)
    mktempdir() do scratch
        write(joinpath(scratch, "input.txt"), "unrelated input")
        mkdir(joinpath(scratch, "temp_user_checkpoint"))
        write(joinpath(scratch, "temp_user_checkpoint", "state.txt"), "unrelated state")
        f(scratch)
        @test read(joinpath(scratch, "input.txt"), String) == "unrelated input"
        @test read(joinpath(scratch, "temp_user_checkpoint", "state.txt"), String) == "unrelated state"
    end
end

function test_cleanup_state(probe, scratch)
    @test probe.initialized == 1
    @test probe.finalized == 1
    @test probe.cleaned == 1
    @test probe.disk isa SUNDMRG.JLD2InternalStorage
    if probe.cleanup_fails
        @test isdir(SUNDMRG._storage_dir(probe.disk))
        @test sort(readdir(scratch)) == sort([
            "input.txt", "temp_user_checkpoint", basename(SUNDMRG._storage_dir(probe.disk)),
        ])
    else
        @test !ispath(SUNDMRG._storage_dir(probe.disk))
        @test sort(readdir(scratch)) == ["input.txt", "temp_user_checkpoint"]
    end
    @test MPI.Initialized() && !MPI.Finalized()
    @test MPI.Allreduce(1, +, MPI.COMM_WORLD) == MPI.Comm_size(MPI.COMM_WORLD)
end

@testset "Runtime cleanup after calculation failures" begin
    MPI.Initialized() || SUNDMRG.init_DMRG!()

    for stage in (:after_storage, :initial_output, :sweep), cleanup_fails in (false, true)
        @testset "$stage, storage cleanup fails=$cleanup_fails" begin
            with_scratch() do scratch
                probe = FailureProbe(; stage, cleanup_fails)
                err = run_probe(probe, scratch)

                @test probe.failure_operation == stage
                expected = cleanup_fails ? [probe.calculation_error, probe.cleanup_error] : [probe.calculation_error]
                @test error_leaves(err) == expected
                if cleanup_fails
                    @test err isa CompositeException
                    @test occursin(probe.calculation_error.msg, sprint(showerror, err))
                else
                    @test err === probe.calculation_error
                end
                test_reported_cleanup_errors(probe, expected[2:end])
                if stage == :after_storage
                    @test isempty(probe.failure_files)
                elseif stage == :initial_output
                    @test probe.failure_files == ["block_l_0.jld2"]
                else
                    @test probe.saved_blocks == Set((label, len) for label in (:l, :r) for len in 0:2)
                    @test "block_l_2.jld2" in probe.failure_files
                    @test "block_r_2.jld2" in probe.failure_files
                end
                test_cleanup_state(probe, scratch)
            end
        end
    end

    @testset "Calculation, storage cleanup, and engine finalization all fail" begin
        with_scratch() do scratch
            probe = FailureProbe(; stage = :sweep, cleanup_fails = true, finalize_fails = true)
            err = run_probe(probe, scratch)
            @test probe.failure_operation == :sweep
            @test err isa CompositeException
            @test error_leaves(err) == [probe.calculation_error, probe.cleanup_error, probe.finalize_error]
            @test occursin(probe.calculation_error.msg, sprint(showerror, err))
            test_reported_cleanup_errors(probe, [probe.cleanup_error, probe.finalize_error])
            test_cleanup_state(probe, scratch)
        end
    end
end

@testset "Runtime cleanup after successful calculations" begin
    for (cleanup_fails, finalize_fails) in ((false, false), (true, false), (false, true), (true, true))
        @testset "storage cleanup fails=$cleanup_fails, engine finalization fails=$finalize_fails" begin
            with_scratch() do scratch
                probe = FailureProbe(; cleanup_fails, finalize_fails)
                result = run_probe(probe, scratch)
                @test probe.failure_operation == :none
                if !cleanup_fails && !finalize_fails
                    @test result[1] == 0
                    @test result[2] isa DMRGOutput
                    @test isapprox(last(result[2].energies), -1.5 - sqrt(3); atol = 1e-10, rtol = 0)
                    test_reported_cleanup_errors(probe, [])
                else
                    expected = Any[]
                    cleanup_fails && push!(expected, probe.cleanup_error)
                    finalize_fails && push!(expected, probe.finalize_error)
                    @test error_leaves(result) == expected
                    if length(expected) == 1
                        @test result === only(expected)
                    else
                        @test result isa CompositeException
                    end
                    @test occursin(first(expected).msg, sprint(showerror, result))
                    test_reported_cleanup_errors(probe, expected[2:end])
                end
                test_cleanup_state(probe, scratch)
            end
        end
    end
end

end # module RuntimeFailureTests
