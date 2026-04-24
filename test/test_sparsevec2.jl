@testset "SparseVector2 operations" begin
    x = SUNDMRG.sparsevec2(Int[3, 1], [2.0, 1.0], 5)
    y = SUNDMRG.sparsevec2(Int[1, 2, 3], [4.0, -1.0, -2.0], 5)

    @test x[1] == 1.0
    @test x[2] == 0.0
    @test x[3] == 2.0

    z = x + y
    @test z[1] == 5.0
    @test z[2] == -1.0
    @test z[3] == 0.0

    @test SUNDMRG.dot(x, y) == 2.0 * -2.0 + 1.0 * 4.0

    a = 2.5 * x
    @test a[1] == 2.5
    @test a[3] == 5.0
end


@testset "SparseVector2 constructors/utilities" begin
    z = SUNDMRG.spzeros2(Float64, Int, 7)
    @test z.n == 7
    @test isempty(z.nzind)
    @test isempty(z.nzval)

    unsorted = SUNDMRG.sparsevec2(Int[5, 2, 4], [50.0, 20.0, 40.0], 6)
    @test unsorted.nzind == [2, 4, 5]
    @test unsorted.nzval == [20.0, 40.0, 50.0]

    SUNDMRG.lmul!(0.5, unsorted)
    @test unsorted.nzval == [10.0, 20.0, 25.0]
end
