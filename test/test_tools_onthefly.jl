@testset "On-the-fly SU(2) helper kernels (1..5)" begin
    Nc = 2
    trivial = SUNDMRG.trivialirrep(Val(Nc))
    funda = SUNDMRG.fundamentalirrep(Val(Nc))
    adj = SUNDMRG.adjointirrep(Val(Nc))

    key1 = (trivial, funda, trivial, funda)
    calc1 = SUNDMRG.on_the_fly_calc1(Nc, key1)
    ref1 = SUNDMRG.wigner9ν(key1[1], funda, key1[2], adj, trivial, adj, key1[3], funda, key1[4])[:, 1, 1, 1, 1, :]
    @test size(calc1) == size(ref1)
    @test calc1 ≈ ref1

    key2 = (trivial, funda, funda)
    calc2 = SUNDMRG.on_the_fly_calc2(Nc, key2)
    ref2 = SUNDMRG.wigner6ν(key2[1], funda, key2[2], adj, key2[3], funda)[1, 1, 1, :]
    @test size(calc2) == size(ref2)
    @test calc2 ≈ ref2

    key3 = (trivial, funda, trivial)
    calc3 = SUNDMRG.on_the_fly_calc3(Nc, key3)
    ref3 = SUNDMRG.wigner9ν(key3[1], funda, key3[2], adj, adj, trivial, key3[3], funda, key3[2])[:, 1, 1, 1, 1, 1]
    @test size(calc3) == size(ref3)
    @test calc3 ≈ ref3

    key4 = (funda, funda, trivial, funda, funda)
    calc4 = SUNDMRG.on_the_fly_calc4(Nc, key4)
    ref4 = SUNDMRG.wigner9ν(key4[1], key4[2], key4[3], adj, adj, trivial, key4[4], key4[5], key4[3])[:, :, :, :, 1, 1]
    @test size(calc4) == size(ref4)
    @test calc4 ≈ ref4

    key5 = (trivial, funda, trivial, trivial, funda)
    calc5 = SUNDMRG.on_the_fly_calc5(Nc, key5)
    ref5 = SUNDMRG.wigner6ν(key5[1], funda, key5[2], key5[3], key5[4], key5[5])[1, :, 1, :] .* SUNDMRG.wigner3ν(key5[3], funda, key5[5])[1, 1]
    @test size(calc5) == size(ref5)
    @test calc5 ≈ ref5

    # Basic SU(2) symmetry/sanity: real-valued kernels and finite entries.
    for arr in (calc1, calc2, calc3, calc4, calc5)
        @test all(isfinite, arr)
        @test all(x -> iszero(imag(x)), arr)
    end
end
