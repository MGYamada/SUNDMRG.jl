@testset "Small SU(2) table-related helpers" begin
    trivial = SUNDMRG.trivialirrep(Val(2))
    funda = SUNDMRG.fundamentalirrep(Val(2))
    adj = SUNDMRG.adjointirrep(Val(2))

    c3 = SUNDMRG._3ν(funda, funda, adj)
    @test size(c3) == (1, 1)

    c6 = SUNDMRG._6ν(funda, funda, adj, funda, funda, adj)
    c6r = SUNDMRG._6νrev(funda, funda, adj, funda, funda, adj)
    @test size(c6) == (1, 1, 1, 1)
    @test size(c6r) == (1, 1, 1, 1)

    c9 = SUNDMRG._9ν(trivial, funda, funda, funda, trivial, funda, trivial, funda, trivial)
    @test size(c9) == (1, 1, 1, 1, 1, 1)

    # On-the-fly table helper matches direct 3ν exchange matrix.
    @test SUNDMRG.on_the_fly_calc6((funda, funda, adj)) == SUNDMRG.wigner3ν(funda, funda, adj)
end
