using Test
using SUNDMRG
using Random

const MPI = SUNDMRG.MPI
const RT = SUNDMRG.RepresentationTheory

function caught_exception(f)
    try
        f()
        return nothing
    catch err
        return err
    end
end

function assert_mpi_usable()
    @test MPI.Initialized()
    @test !MPI.Finalized()
    @test MPI.Comm_rank(MPI.COMM_WORLD) == 0
    @test MPI.Comm_size(MPI.COMM_WORLD) == 1
    @test MPI.Allreduce(17, +, MPI.COMM_WORLD) == 17
end

function assert_owned_mpi_finalized()
    @test MPI.Initialized()
    @test MPI.Finalized()
    @test !finalize_DMRG!()
    @test_throws r"MPI has already been finalized" init_DMRG!()
    @test_throws r"MPI has already been finalized" RT._init_table_mpi!(true)
end

function small_dmrg(; manage_mpi, kwargs...)
    Random.seed!(1858)
    return run_DMRG(
        SU(2)HeisenbergModel(), SquareLattice(2, 2), 8, [8], 8, CPUEngine;
        verbose = false, manage_mpi = manage_mpi, kwargs...,
    )
end

function failing_dmrg(; manage_mpi)
    Random.seed!(1858)
    return run_DMRG(
        SU(2)HeisenbergModel(), SquareLattice(4, 2), 1, [1], 20, CPUEngine;
        verbose = false, manage_mpi = manage_mpi, fileio = true,
        scratch = pwd(), max_cooldown_sweeps = 1, tol_energy = 1e-10, tol_EE = 1e-10,
    )
end

function check_dmrg_output(output)
    rank, result = output
    @test rank == 0
    @test last(result.energies) ≈ -1.5 - sqrt(3) atol = 1e-10 rtol = 0
end

function check_table(builder, filename; manage_mpi)
    @test builder(2, 0; manage_mpi = manage_mpi) === nothing
    @test isfile(filename)
    table = SUNDMRG.RepresentationTheory.JLD2.load(filename)
    @test length(table) == 1
    @test only(values(table)) isa Dict
end

function fail_table_output(builder, filename; manage_mpi)
    # An existing directory at the required output path fails on all supported
    # platforms, including users for whom read-only permissions do not apply.
    mktempdir() do workdir
        cd(workdir) do
            mkdir(filename)
            err = caught_exception() do
                builder(2, 0; manage_mpi = manage_mpi)
            end
            @test err !== nothing
            @test occursin(filename, sprint(showerror, err))
            @test isdir(filename)
        end
    end
end

function scenario_caller_owned()
    # Public argument validation must not acquire MPI. Caller-managed calls
    # must reject an uninitialized process without creating MPI state either.
    @test !MPI.Initialized()
    @test !MPI.Finalized()
    @test_throws ArgumentError small_dmrg(; manage_mpi = true, alg = :invalid)
    @test_throws ArgumentError small_dmrg(; manage_mpi = false)
    for builder in (make_table3nu, make_table4)
        @test_throws ArgumentError builder(1, 0)
        @test_throws ArgumentError builder(2, -1)
        @test_throws ArgumentError builder(2, 0; manage_mpi = :invalid)
        @test_throws ArgumentError builder(2, 0; manage_mpi = false)
    end
    @test !MPI.Initialized()
    @test !MPI.Finalized()

    @test init_DMRG!()
    try
        # Default management must also leave MPI alive when the caller owns it.
        for manage_mpi in (false, false, true)
            check_dmrg_output(small_dmrg(; manage_mpi = manage_mpi))
            assert_mpi_usable()
        end
        for manage_mpi in (false, true)
            before = Set(readdir())
            @test_throws r"DMRG did not converge within 1 cooldown sweeps" failing_dmrg(; manage_mpi = manage_mpi)
            @test Set(readdir()) == before
            assert_mpi_usable()
        end
        for (builder, filename) in (
            (make_table3nu, "table3nuhalf_SU2_0.jld2"),
            (make_table4, "table4half_SU2_0.jld2"),
        )
            for manage_mpi in (false, false, true)
                check_table(builder, filename; manage_mpi = manage_mpi)
                assert_mpi_usable()
            end
            for manage_mpi in (false, true)
                fail_table_output(builder, filename; manage_mpi = manage_mpi)
                assert_mpi_usable()
            end
        end
        # Exceptions must not leave communicators or other runtime state that
        # prevents a later calculation in the caller's long-lived MPI process.
        check_dmrg_output(small_dmrg(; manage_mpi = false))
        assert_mpi_usable()
    finally
        finalize_DMRG!()
    end
    assert_owned_mpi_finalized()
end

function scenario_dmrg_owned(successful)
    @test !MPI.Initialized()
    before = Set(readdir())
    if successful
        check_dmrg_output(small_dmrg(; manage_mpi = true, fileio = true, scratch = pwd()))
    else
        @test_throws r"DMRG did not converge within 1 cooldown sweeps" failing_dmrg(; manage_mpi = true)
    end
    @test Set(readdir()) == before
    assert_owned_mpi_finalized()
end

function scenario_table_owned(builder, filename, successful)
    @test !MPI.Initialized()
    if successful
        check_table(builder, filename; manage_mpi = true)
    else
        fail_table_output(builder, filename; manage_mpi = true)
    end
    assert_owned_mpi_finalized()
end

function scenario_table_partial_init_failure()
    # MPI.Init runs initialization hooks after acquiring MPI. A failing hook
    # must still release MPI, although the initializer never returned ownership.
    MPI.add_init_hook!(() -> error("injected error after MPI initialization"))
    @test !MPI.Initialized()
    invoked = Ref(false)
    operation() = RT._with_table_mpi(true) do
        invoked[] = true
    end
    @test_throws r"injected error after MPI initialization" operation()
    @test !invoked[]
    assert_owned_mpi_finalized()
end

function scenario_dmrg_partial_init_failure()
    MPI.add_init_hook!(() -> error("injected error after MPI initialization"))
    @test !MPI.Initialized()
    @test_throws r"injected error after MPI initialization" small_dmrg(; manage_mpi = true)
    assert_owned_mpi_finalized()
end

function scenario_dmrg_cleanup_failure(calculation_fails)
    @eval SUNDMRG begin
        const _test_engine_finalize_calls = Ref(0)
        function _finalize_engine_runtime!(::Type{CPUEngine})
            _test_engine_finalize_calls[] += 1
            error("injected engine cleanup failure")
        end
    end
    operation() = calculation_fails ? failing_dmrg(; manage_mpi = true) :
        small_dmrg(; manage_mpi = true, fileio = true, scratch = pwd())
    before = Set(readdir())
    if calculation_fails
        actual = @test_logs (:error, "Cleanup failed while handling an earlier exception") caught_exception(() -> Base.invokelatest(operation))
        @test actual isa CompositeException
        failures = actual.exceptions
        @test length(failures) == 2
        @test failures[1] isa CapturedException
        @test occursin("DMRG did not converge within 1 cooldown sweeps", sprint(showerror, failures[1]))
        @test failures[2] isa CapturedException
        @test occursin("injected engine cleanup failure", sprint(showerror, failures[2]))
    else
        @test_throws r"injected engine cleanup failure" Base.invokelatest(operation)
    end
    @test SUNDMRG._test_engine_finalize_calls[] == 1
    @test Set(readdir()) == before
    assert_owned_mpi_finalized()
end

function scenario_table_cleanup_failure(calculation_fails)
    # This replacement is confined to this worker. Finalization really occurs;
    # then an injected error checks exception precedence and exactly-once cleanup.
    @eval RT begin
        const _test_finalize_calls = Ref(0)
        function _finalize_table_mpi!(owned::Bool)
            owned || return false
            _test_finalize_calls[] += 1
            MPI.Finalize()
            error("injected MPI cleanup failure")
        end
    end
    original = ErrorException("injected coefficient calculation failure")
    operation() = RT._with_table_mpi(true) do
        assert_mpi_usable()
        calculation_fails && throw(original)
        return :ok
    end
    if calculation_fails
        actual = @test_logs (:error, "Cleanup failed while handling an earlier exception") caught_exception(() -> Base.invokelatest(operation))
        @test actual isa CompositeException
        failures = actual.exceptions
        @test length(failures) == 2
        @test failures[1] isa CapturedException
        @test failures[1].ex === original
        @test failures[2] isa CapturedException
        @test occursin("injected MPI cleanup failure", sprint(showerror, failures[2]))
    else
        @test_throws r"injected MPI cleanup failure" Base.invokelatest(operation)
    end
    @test RT._test_finalize_calls[] == 1
    assert_owned_mpi_finalized()
end

scenario = only(ARGS)
@testset "MPI ownership: $scenario" begin
    if scenario == "caller-owned"
        scenario_caller_owned()
    elseif scenario == "dmrg-owned-success"
        scenario_dmrg_owned(true)
    elseif scenario == "dmrg-owned-failure"
        scenario_dmrg_owned(false)
    elseif scenario == "dmrg-partial-init-failure"
        scenario_dmrg_partial_init_failure()
    elseif scenario == "dmrg-engine-cleanup-failure"
        scenario_dmrg_cleanup_failure(false)
    elseif scenario == "dmrg-calculation-and-cleanup-failure"
        scenario_dmrg_cleanup_failure(true)
    elseif scenario == "table3nu-owned-success"
        scenario_table_owned(make_table3nu, "table3nuhalf_SU2_0.jld2", true)
    elseif scenario == "table3nu-owned-failure"
        scenario_table_owned(make_table3nu, "table3nuhalf_SU2_0.jld2", false)
    elseif scenario == "table4-owned-success"
        scenario_table_owned(make_table4, "table4half_SU2_0.jld2", true)
    elseif scenario == "table4-owned-failure"
        scenario_table_owned(make_table4, "table4half_SU2_0.jld2", false)
    elseif scenario == "table-partial-init-failure"
        scenario_table_partial_init_failure()
    elseif scenario == "table-cleanup-failure"
        scenario_table_cleanup_failure(false)
    elseif scenario == "table-calculation-and-cleanup-failure"
        scenario_table_cleanup_failure(true)
    else
        error("Unknown ownership scenario: $scenario")
    end
end
println("MPI_OWNERSHIP_OK $scenario")
