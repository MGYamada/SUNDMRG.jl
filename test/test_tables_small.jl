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

@testset "SU(2) 3ν/6ν/9ν coefficient values" begin
    trivial = SUNDMRG.trivialirrep(Val(2))
    funda = SUNDMRG.fundamentalirrep(Val(2))
    adj = SUNDMRG.adjointirrep(Val(2))

    c3 = SUNDMRG._3ν(funda, funda, adj)
    w3 = SUNDMRG.wigner3ν(funda, funda, adj)
    @test c3 ≈ w3
    @test only(c3) ≈ 1.0

    c6 = SUNDMRG._6ν(funda, funda, adj, funda, funda, adj)
    w6 = SUNDMRG.wigner6ν(funda, funda, adj, funda, funda, adj)
    @test c6 ≈ w6
    @test only(c6) ≈ 0.5

    c6r = SUNDMRG._6νrev(funda, funda, adj, funda, funda, adj)
    @test c6r ≈ permutedims(c6, (3, 4, 1, 2))

    nineν_cases = (
        (trivial, funda, funda, adj, trivial, adj, adj, funda, funda),
        (funda, funda, trivial, adj, adj, trivial, funda, funda, trivial),
        (funda, funda, adj, funda, funda, adj, adj, adj, trivial),
    )
    expected = (-1.0, 1 / sqrt(3), -0.5)
    for (case, value) in zip(nineν_cases, expected)
        c9 = SUNDMRG._9ν(case...)
        w9 = SUNDMRG.wigner9ν(case...)
        @test c9 ≈ w9
        @test only(c9) ≈ value
    end
end

@testset "RepresentationTheory submodule coefficient access" begin
    RT = SUNDMRG.RepresentationTheory
    trivial = RT.trivialirrep(Val(2))
    funda = RT.fundamentalirrep(Val(2))
    adj = RT.adjointirrep(Val(2))

    @test RT._3ν(funda, funda, adj) ≈ SUNDMRG._3ν(funda, funda, adj)
    @test RT._6ν(funda, funda, adj, funda, funda, adj) ≈ SUNDMRG._6ν(funda, funda, adj, funda, funda, adj)
    @test RT._9ν(trivial, funda, funda, adj, trivial, adj, adj, funda, funda) ≈ SUNDMRG._9ν(trivial, funda, funda, adj, trivial, adj, adj, funda, funda)
end
