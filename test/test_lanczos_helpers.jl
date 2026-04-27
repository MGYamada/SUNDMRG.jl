using LinearAlgebra

@testset "Lanczos nested-array helper kernels" begin
    function nested_fixture(offset::Float64)
        z = Matrix{Vector{Matrix{Float64}}}(undef, 2, 2)
        z[1, 1] = [reshape(collect(1.0:4.0) .+ offset, 2, 2), [5.0 + offset 6.0 + offset; 7.0 + offset 8.0 + offset]]
        z[1, 2] = [[9.0 + offset 10.0 + offset; 11.0 + offset 12.0 + offset]]
        z[2, 1] = [[13.0 + offset 14.0 + offset; 15.0 + offset 16.0 + offset], [17.0 + offset 18.0 + offset; 19.0 + offset 20.0 + offset]]
        z[2, 2] = [[21.0 + offset 22.0 + offset; 23.0 + offset 24.0 + offset]]
        z
    end

    x = nested_fixture(0.0)
    y = nested_fixture(0.5)

    manual_dot = 0.0
    for I in eachindex(x), J in eachindex(x[I])
        manual_dot += LinearAlgebra.dot(x[I][J], y[I][J])
    end
    @test SUNDMRG.mydot(x, y) ≈ manual_dot
    @test SUNDMRG.mydot(x, y) ≈ SUNDMRG.mydot(y, x)

    y_axpy = deepcopy(y)
    SUNDMRG.myaxpy!(2.0, x, y_axpy)
    for I in eachindex(x), J in eachindex(x[I])
        @test y_axpy[I][J] ≈ y[I][J] .+ 2.0 .* x[I][J]
    end

    y_axpby = deepcopy(y)
    SUNDMRG.myaxpby!(1.5, x, -0.25, y_axpby)
    for I in eachindex(x), J in eachindex(x[I])
        @test y_axpby[I][J] ≈ 1.5 .* x[I][J] .- 0.25 .* y[I][J]
    end

    dest = nested_fixture(-100.0)
    SUNDMRG.mycopyto!(dest, x)
    @test dest == x

    # Copy must be value-based, not aliasing source storage.
    x[1, 1][1][1, 1] = -999.0
    @test dest[1, 1][1][1, 1] != x[1, 1][1][1, 1]
end
