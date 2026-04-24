using Test
using SUNDMRG

@testset "SUNDMRG CPU-only pure tests" begin
    include("test_init.jl")
    include("test_suncalc.jl")
    include("test_sparsevec2.jl")
    include("test_tables_small.jl")
end
