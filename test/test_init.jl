@testset "Model/lattice/engine initialization" begin
    model = SU(2)HeisenbergModel()
    @test model isa SUNDMRG.HeisenbergModelSU{2}

    sq = SquareLattice(4, 2)
    @test sq.Lx == 4
    @test sq.Ly == 2

    hc = HoneycombLattice(4, 2, :ZC)
    @test hc.Lx == 4
    @test hc.Ly == 2
    @test hc.BC == :ZC

    @test_throws AssertionError HoneycombLattice(4, 3, :ZC)
    @test_throws AssertionError HoneycombLattice(4, 2, :PBC)

    left = SUNDMRG.Block(2, Tuple{Int, Int}[], [SUNDMRG.trivialirrep(Val(2))], [1], [1], Dict(:H => [zeros(1, 1)]))
    right = SUNDMRG.Block(3, Tuple{Int, Int}[], [SUNDMRG.trivialirrep(Val(2))], [1], [1], Dict(:H => [zeros(1, 1)]))

    @test SUNDMRG.graphic(left, right) == "==**---"
    @test SUNDMRG.graphic(left, right; sys_label = :r) == "---**=="
    @test_throws ArgumentError SUNDMRG.graphic(left, right; sys_label = :x)
end
