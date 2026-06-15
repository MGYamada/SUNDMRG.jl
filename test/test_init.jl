@testset "Model/lattice/engine initialization" begin
    model = SUNDMRG.SU(2) * SUNDMRG.HeisenbergModel()
    @test model isa SUNDMRG.HeisenbergModelSU{2}

    sq = SUNDMRG.SquareLattice(4, 2)
    @test sq.Lx == 4
    @test sq.Ly == 2

    hc = SUNDMRG.HoneycombLattice(4, 2, :ZC)
    @test hc.Lx == 4
    @test hc.Ly == 2
    @test hc.BC == :ZC

    @test_throws ArgumentError SUNDMRG.HoneycombLattice(4, 3, :ZC)
    @test_throws ArgumentError SUNDMRG.HoneycombLattice(4, 2, :PBC)

    left = (length = 2,)
    right = (length = 3,)

    @test SUNDMRG.graphic(left, right) == "==**---"
    @test SUNDMRG.graphic(left, right; sys_label = :r) == "---**=="
    @test_throws ArgumentError SUNDMRG.graphic(left, right; sys_label = :x)
end

@testset "DMRG schedule normalization" begin
    @test SUNDMRG._dmrg_schedule(12) == (12, 0.0)
    @test SUNDMRG._dmrg_schedule((12, 0.25)) == (12, 0.25)

    schedules = SUNDMRG._dmrg_schedule_list(Any[4, (8, 1e-3), 16])
    @test schedules == [(4, 0.0), (8, 1e-3), (16, 0.0)]
    @test schedules isa Vector{Tuple{Int, Float64}}

    @test SUNDMRG._dmrg_schedule((12, 0)) == (12, 0.0)
    @test SUNDMRG._dmrg_schedule_list([(4, 0), (8, 1)]) == [(4, 0.0), (8, 1.0)]

    @test_throws ArgumentError SUNDMRG._dmrg_schedule(0)
    @test_throws ArgumentError SUNDMRG._dmrg_schedule((-1, 0.0))
    @test_throws ArgumentError SUNDMRG._dmrg_schedule((12, -1e-3))
    @test_throws ArgumentError SUNDMRG._dmrg_schedule((12, Inf))
    @test_throws ArgumentError SUNDMRG._dmrg_schedule((12, NaN))
end

@testset "Runtime validation helpers" begin
    on_the_fly, mirror, γ_type, γ_list, N, signfactor = SUNDMRG._init_runtime_and_engine(SUNDMRG.CPUEngine, :square, 4, 4, 2, 0, 1)

    @test on_the_fly == Val(true)
    @test mirror
    @test γ_type == typeof(SUNDMRG.trivialirrep(Val(2)))
    @test length(γ_list) == 2
    @test N == 16
    @test signfactor == -1.0

    @test_throws ArgumentError SUNDMRG._init_runtime_and_engine(SUNDMRG.CPUEngine, :square, 4, 4, 1, 0, 1)
    @test_throws ArgumentError SUNDMRG._init_runtime_and_engine(SUNDMRG.CPUEngine, :triangular, 4, 4, 2, 0, 1)
    @test_throws ArgumentError SUNDMRG._init_runtime_and_engine(SUNDMRG.CPUEngine, :square, 4, 4, 3, 0, 1)
    @test_throws ArgumentError SUNDMRG._init_runtime_and_engine(SUNDMRG.CPUEngine, :square, 4, 3, 4, 0, 1)
end
