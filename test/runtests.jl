using Test
using SUNDMRG

@testset "SUNDMRG CPU tests" begin
    include("test_init.jl")
    include("test_suncalc.jl")
    include("test_sparsevec2.jl")
    include("test_storage.jl")
    include("test_magma_runtime.jl")
    include("test_finite_helpers.jl")
    include("test_tables_small.jl")
    include("test_tables_ground_truth.jl")
    include("test_representation_theory_internal.jl")
    include("test_representation_theory_definitions.jl")
    include("test_sun_ed.jl")
    include("test_step_helpers.jl")
    include("test_lanczos_helpers.jl")
    include("test_lanczos_multiplicity.jl")
    include("test_tools_onthefly.jl")
    include("test_wigner_racah_definitions.jl")
    include("test_numerical_boundaries.jl")
    include("test_runtime_failures.jl")
    include("test_run_dmrg.jl")
    include("test_mpi_ownership.jl")
end
