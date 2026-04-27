@testset "dcinit partition helper" begin
    @test SUNDMRG.dcinit(0, 4) == [0, 0, 0, 0, 0]

    # Edge case: fewer tasks than workers.
    dc_small = SUNDMRG.dcinit(3, 5)
    @test length(dc_small) == 6
    @test dc_small == [0, 0, 1, 1, 2, 3]
    @test diff(dc_small) == [0, 1, 0, 1, 1]

    # Divisibility case: perfectly even partition.
    dc_even = SUNDMRG.dcinit(12, 4)
    @test dc_even == [0, 3, 6, 9, 12]
    @test all(==(3), diff(dc_even))

    # General monotonic / coverage sanity.
    for (N, Ncpu) in ((1, 3), (7, 4), (13, 6), (64, 8))
        dc = SUNDMRG.dcinit(N, Ncpu)
        @test first(dc) == 0
        @test last(dc) == N
        @test issorted(dc)
        @test all(x -> x in (fld(N, Ncpu), cld(N, Ncpu)), diff(dc))
        @test maximum(diff(dc)) - minimum(diff(dc)) <= 1
    end
end
