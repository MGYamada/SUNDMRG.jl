module MPIFailureScenarios

using Test
using Random
using TOML
using MPI
using SUNDMRG

Base.@kwdef mutable struct FailureProbe
    stage::Symbol
    rank::Int = -1
    nranks::Int = 0
    initialized::Int = 0
    finalized::Int = 0
    cleaned::Int = 0
    finalized_while_mpi_live::Bool = false
    disk::Union{Nothing, SUNDMRG.JLD2InternalStorage} = nothing
    saved_blocks::Set{Tuple{Symbol, Int}} = Set{Tuple{Symbol, Int}}()
    fault_path::String = ""
    original_error::Any = nothing
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

function SUNDMRG._init_engine_runtime!(::Type{TrackingCPUEngine}, rank, nranks)
    probe = active_probe[]::FailureProbe
    probe.rank = rank
    probe.nranks = nranks
    probe.initialized += 1
    return nothing
end

function SUNDMRG._finalize_engine_runtime!(::Type{TrackingCPUEngine})
    probe = active_probe[]::FailureProbe
    probe.finalized += 1
    probe.finalized_while_mpi_live = MPI.Initialized() && !MPI.Finalized()
    return nothing
end

function prepare_scratch(scratch)
    mkpath(scratch)
    write(joinpath(scratch, "input.txt"), "unrelated input")
    mkpath(joinpath(scratch, "temp_user_checkpoint"))
    write(joinpath(scratch, "temp_user_checkpoint", "state.txt"), "unrelated state")
    return nothing
end

function SUNDMRG.init_internal_storage(fileio::Val{true}, scratch::FailureScratch,
        block_table, trmat_table, tensor_table, rank)
    # Package-owned jobs cannot call MPI first just to determine their rank.
    # Only root prepares the shared scratch directory at this existing seam.
    if rank == 0 && !isdir(scratch.path)
        prepare_scratch(scratch.path)
    end
    disk = SUNDMRG.init_internal_storage(fileio, scratch.path,
        block_table, trmat_table, tensor_table, rank)
    scratch.probe.disk = disk
    return FailureStorage(disk, scratch.probe)
end

function fail_write!(write_file, storage, filename)
    probe = storage.probe
    probe.rank == 0 || error("the I/O fixture must only modify files on rank 0")
    isempty(probe.fault_path) || error("the I/O fixture must fail only once")
    probe.fault_path = filename
    # Make the actual JLD2 writer fail on a directory in place of its output
    # file. This leaves a partial package directory for real cleanup to remove.
    ispath(filename) && rm(filename; recursive = true)
    mkdir(filename)
    try
        write_file()
    catch err
        probe.original_error = err
        rethrow()
    end
    error("expected JLD2 to reject a directory as an output file")
end

function SUNDMRG.save_block(storage::FailureStorage, label, len, block)
    probe = storage.probe
    # With a 2x2 lattice, warmup writes l/r lengths 0,1,2 once and growth is
    # unnecessary. The first overwrite is right length 1 in the first sweep.
    if probe.stage == :sweep_output && (label, len) in probe.saved_blocks
        return fail_write!(storage, SUNDMRG._block_filename(storage.disk, label, len)) do
            SUNDMRG.save_block(storage.disk, label, len, block)
        end
    end
    SUNDMRG.save_block(storage.disk, label, len, block)
    push!(probe.saved_blocks, (label, len))
    return nothing
end

function SUNDMRG.save_trmat(storage::FailureStorage, label, len, trmat)
    if storage.probe.stage == :initial_output && label == :l && len == 0
        return fail_write!(storage, SUNDMRG._trmat_filename(storage.disk, label, len)) do
            SUNDMRG.save_trmat(storage.disk, label, len, trmat)
        end
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
    SUNDMRG.cleanup_storage!(storage.disk)
    if storage.probe.stage == :cleanup
        # A finalizer may remove its resource and then fail. All peers must
        # still receive the failure and finalize their engine exactly once.
        storage.probe.original_error = ErrorException("injected root-only storage cleanup failure")
        throw(storage.probe.original_error)
    end
    return nothing
end

function capture_run(probe, scratch; manage_mpi, invalid_input = false, invalid_rank0 = false)
    previous_probe = active_probe[]
    active_probe[] = probe
    try
        Random.seed!(1858)
        lattice = SquareLattice(invalid_input ? 3 : 2, 2)
        tolerance = invalid_rank0 && probe.rank == 0 ? 0.0 : 1e-5
        return run_DMRG(SU(2) * HeisenbergModel(), lattice, 8, [8], 8, TrackingCPUEngine;
            fileio = true, scratch = FailureScratch(scratch, probe), tol_energy = tolerance,
            max_cooldown_sweeps = 8, verbose = false, manage_mpi)
    catch err
        return err
    finally
        active_probe[] = previous_probe
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

function test_scratch(scratch)
    @test read(joinpath(scratch, "input.txt"), String) == "unrelated input"
    @test read(joinpath(scratch, "temp_user_checkpoint", "state.txt"), String) == "unrelated state"
    @test sort(readdir(scratch)) == ["input.txt", "temp_user_checkpoint"]
end

function test_probe(probe, scratch; invalid = false)
    @test probe.rank >= 0
    @test probe.nranks >= 1
    @test probe.initialized == (invalid ? 0 : 1)
    @test probe.finalized == (invalid ? 0 : 1)
    @test probe.cleaned == (!invalid && probe.rank == 0 ? 1 : 0)
    if invalid
        @test probe.disk === nothing
    else
        @test probe.finalized_while_mpi_live
        @test probe.disk isa SUNDMRG.JLD2InternalStorage
        if probe.rank == 0
            @test !ispath(SUNDMRG._storage_dir(probe.disk))
            @test probe.original_error isa Exception
            if probe.stage == :cleanup
                @test isempty(probe.fault_path)
            elseif probe.stage == :sweep_output
                @test basename(probe.fault_path) == "block_r_1.jld2"
                @test probe.saved_blocks == Set((label, len) for label in (:l, :r) for len in 0:2)
            else
                @test basename(probe.fault_path) == "trmat_l_0.jld2"
                @test probe.saved_blocks == Set([(:l, 0)])
            end
        else
            @test isempty(probe.fault_path)
            @test isempty(probe.saved_blocks)
            @test probe.original_error === nothing
        end
    end
    probe.rank == 0 && test_scratch(scratch)
end

function check_error(result, probe, expected_message; propagated = true)
    @test result isa Exception
    message = result isa Exception ? sprint(showerror, result) : ""
    @test occursin(expected_message, message)
    if propagated && probe.rank != 0
        @test occursin("rank 0", message)
    end
    if probe.original_error !== nothing
        @test any(err -> err === probe.original_error, error_leaves(result))
    end
    return message
end

function run_caller_scenario(scenario, scratch)
    @test SUNDMRG.init_DMRG!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)
    rank == 0 && prepare_scratch(scratch)
    MPI.Barrier(comm)

    invalid_input = scenario == "caller_invalid_input"
    invalid_rank0 = scenario == "caller_invalid_rank0"
    invalid = invalid_input || invalid_rank0
    stage = scenario == "caller_sweep_output" ? :sweep_output :
        scenario == "caller_cleanup" ? :cleanup : :initial_output
    probe = FailureProbe(; stage, rank, nranks)
    result = capture_run(probe, scratch; manage_mpi = false, invalid_input, invalid_rank0)

    @test MPI.Initialized() && !MPI.Finalized()
    @test MPI.Allreduce(Int(result isa Exception), +, comm) == nranks
    expected_message = invalid_input ? "Lx must be even" : invalid_rank0 ?
        "tol_energy must be positive and finite" : stage == :cleanup ?
        "injected root-only storage cleanup failure" : stage == :initial_output ?
        "trmat_l_0.jld2" : "block_r_1.jld2"
    message = check_error(result, probe, expected_message; propagated = !invalid_input)
    if !invalid
        original_message = MPI.bcast(rank == 0 ? sprint(showerror, probe.original_error) : nothing, 0, comm)
        @test occursin(original_message, message)
    end
    test_probe(probe, scratch; invalid)

    # Reuse both the same MPI communicator and the same scratch directory after
    # every synchronized exception. The ordinary CPU/JLD2 path must still work.
    @test MPI.Allreduce(rank + 1, +, comm) == nranks * (nranks + 1) ÷ 2
    Random.seed!(1858)
    returned_rank, output = run_DMRG(SU(2) * HeisenbergModel(), SquareLattice(2, 2),
        8, [8], 8, CPUEngine; fileio = true, scratch, max_cooldown_sweeps = 8,
        verbose = false, manage_mpi = false)
    @test returned_rank == rank
    if rank == 0
        @test output isa DMRGOutput
        @test isapprox(last(output.energies), -1.5 - sqrt(3); atol = 1e-10, rtol = 0)
        test_scratch(scratch)
    else
        @test output === nothing
    end
    @test MPI.Allreduce(1, +, comm) == nranks
    @test SUNDMRG.finalize_DMRG!()
    @test MPI.Finalized()

    return Dict("rank" => rank, "nranks" => nranks, "message" => message,
        "mpi_finalized" => true, "caller_reused" => true,
        "initialized" => probe.initialized, "finalized" => probe.finalized,
        "cleaned" => probe.cleaned)
end

function run_owned_scenario(scenario, scratch)
    @test !MPI.Initialized()
    stage = scenario == "owned_sweep_output" ? :sweep_output : :initial_output
    probe = FailureProbe(; stage)
    result = capture_run(probe, scratch; manage_mpi = true)

    @test MPI.Initialized() && MPI.Finalized()
    expected_message = stage == :initial_output ? "trmat_l_0.jld2" : "block_r_1.jld2"
    message = check_error(result, probe, expected_message)
    test_probe(probe, scratch)

    return Dict("rank" => probe.rank, "nranks" => probe.nranks, "message" => message,
        "mpi_finalized" => true, "caller_reused" => false,
        "initialized" => probe.initialized, "finalized" => probe.finalized,
        "cleaned" => probe.cleaned)
end

function main(args)
    length(args) == 2 || error("usage: mpi_failure_scenarios.jl scenario output_dir")
    scenario, output_dir = args
    scenario in ("caller_initial_output", "caller_sweep_output", "caller_cleanup", "caller_invalid_input",
        "caller_invalid_rank0", "owned_initial_output", "owned_sweep_output") ||
        error("unknown MPI failure scenario: $scenario")
    mkpath(output_dir)
    scratch = joinpath(output_dir, "scratch")
    report = nothing
    @testset "MPI failure scenario: $scenario" begin
        report = startswith(scenario, "caller_") ? run_caller_scenario(scenario, scratch) :
            run_owned_scenario(scenario, scratch)
    end
    report["scenario"] = scenario
    open(joinpath(output_dir, "failure-$scenario-rank$(report["rank"]).toml"), "w") do io
        TOML.print(io, report)
    end
    if report["rank"] == 0
        println("MPI_FAILURE_OK $scenario")
    end
    return nothing
end

end # module MPIFailureScenarios

if abspath(PROGRAM_FILE) == @__FILE__
    MPIFailureScenarios.main(ARGS)
end
