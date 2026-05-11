using SUNRepresentations: weight

@testset "Pure SU helper functions" begin
    irr0 = SUNDMRG.irrep(3, 0)
    irr1 = SUNDMRG.irrep(3, 1)
    @test irr0 == [0, 0, 0]
    @test irr1 == [1, 0, 0]

    λ = SUNDMRG.irreplist(2, 3)
    @test first(weight.(λ)) == (0, 0)
    @test last(weight.(λ)) == (3, 0)
    @test length(λ) == 4

    trivial = SUNDMRG.trivialirrep(Val(2))
    funda = SUNDMRG.fundamentalirrep(Val(2))
    adj = SUNDMRG.adjointirrep(Val(2))
    @test weight(trivial) == (0, 0)
    @test weight(funda) == (1, 0)
    @test weight(adj) == (2, 0)

    β = [trivial, funda, adj]
    OM = SUNDMRG.OM_matrix(β, β, trivial)
    @test size(OM) == (3, 3)
    @test OM[1, 1] == 1

    # SU(2) consistency helpers
    w3 = SUNDMRG.wigner3ν(funda, funda, adj)
    @test size(w3) == (1, 1)

    w6 = SUNDMRG.wigner6ν(funda, funda, adj, funda, funda, adj)
    w6r = SUNDMRG.wigner6νrev(funda, funda, adj, funda, funda, adj)
    @test size(w6) == reverse(size(w6r))
end


@testset "Additional irrep/OM consistency" begin
    λ0 = SUNDMRG.irreplist(3, 0)
    @test length(λ0) == 1
    @test weight(only(λ0)) == (0, 0, 0)

    trivial3 = SUNDMRG.trivialirrep(Val(3))
    funda3 = SUNDMRG.fundamentalirrep(Val(3))
    OMself = SUNDMRG.OM_matrix([trivial3, funda3], trivial3)
    @test size(OMself) == (2, 2)
    @test OMself[1, 1] == 1
    @test_throws ArgumentError SUNDMRG.wigner9ν(trivial3, trivial3, trivial3, trivial3, trivial3, trivial3, trivial3, trivial3, trivial3)
end
