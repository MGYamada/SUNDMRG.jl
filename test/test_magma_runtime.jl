module MAGMARuntimeTests

using Test
using Logging
using SUNDMRG

function caught(f)
    try
        f()
    catch error
        return error
    end
    return nothing
end

@testset "MAGMA reference ownership with simulated C API statuses" begin
    # The C library counts every returned init call, including error statuses.
    # Model this boundary without loading MAGMA or requiring a GPU in CPU CI.
    @testset "Successful initialization preserves existing references" begin
        for existing_references in (0, 2)
            references = Ref(existing_references)
            finalizations = Ref(0)
            initialize() = (references[] += 1; Int32(0))
            finalize() = (references[] -= 1; finalizations[] += 1; Int32(0))

            @test SUNDMRG._init_magma_runtime!(initialize, finalize) === nothing
            @test references[] == existing_references + 1
            @test finalizations[] == 0
            @test SUNDMRG._finalize_magma_runtime!(finalize) === nothing
            @test references[] == existing_references
            @test finalizations[] == 1
        end
    end

    @testset "Failed initialization rolls back exactly one reference" begin
        for init_status in (Int32(-112), Int32(-116)), final_status in (Int32(0), Int32(-101))
            references = Ref(2)
            finalizations = Ref(0)
            initialize() = (references[] += 1; init_status)
            finalize() = (references[] -= 1; finalizations[] += 1; final_status)
            logger = Test.TestLogger()
            error = with_logger(logger) do
                caught(() -> SUNDMRG._init_magma_runtime!(initialize, finalize))
            end

            @test references[] == 2
            @test finalizations[] == 1
            @test occursin("magma_init failed with status $init_status", sprint(showerror, error))
            if final_status == 0
                @test error isa ErrorException
                @test isempty(logger.logs)
            else
                @test error isa CompositeException
                @test length(error.exceptions) == 2
                @test all(e -> e isa CapturedException && !isempty(e.processed_bt), error.exceptions)
                @test length(logger.logs) == 1
                logged_error, backtrace = only(logger.logs).kwargs[:exception]
                @test logged_error === error.exceptions[2].ex
                @test occursin("magma_finalize failed with status $final_status", sprint(showerror, logged_error))
                @test !isempty(backtrace)
            end
        end
    end

    @testset "A library-call exception transfers no reference" begin
        library_error = ErrorException("simulated library loading failure")
        finalizations = Ref(0)
        initialize() = throw(library_error)
        finalize() = (finalizations[] += 1; Int32(0))
        @test caught(() -> SUNDMRG._init_magma_runtime!(initialize, finalize)) === library_error
        @test finalizations[] == 0
    end

    @testset "A failed finalizer is not retried" begin
        for throw_exception in (false, true)
            finalizations = Ref(0)
            finalization_error = ErrorException("simulated finalization call failure")
            finalize() = begin
                finalizations[] += 1
                throw_exception && throw(finalization_error)
                return Int32(-101)
            end
            error = caught(() -> SUNDMRG._finalize_magma_runtime!(finalize))
            @test finalizations[] == 1
            if throw_exception
                @test error === finalization_error
            else
                @test occursin("magma_finalize failed with status -101", sprint(showerror, error))
            end
        end
    end
end

end # module MAGMARuntimeTests
