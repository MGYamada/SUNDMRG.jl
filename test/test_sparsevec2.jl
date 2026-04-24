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
