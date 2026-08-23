using LinearAlgebra
using MPI
using Random

MPI.Initialized() || MPI.Init()

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

    SUNDMRG.myzero!(dest)
    for I in eachindex(dest), J in eachindex(dest[I])
        @test iszero(dest[I][J])
    end
end

@testset "Lanczos clears output before applying accumulating operator" begin
    H = Diagonal([1.0, 2.0])
    initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
    initial[1, 1] = [reshape([1.0, 1.0], 2, 1)]

    function accumulating_A!(out, input)
        out[1, 1][1] .+= H * input[1, 1][1]
        return out
    end

    val = SUNDMRG.Lanczos!(accumulating_A!, initial, 1, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 4, alg = :slow)
    @test val ≈ 1.0 atol = 1e-12
end

@testset "Lanczos resolves distinct excited states" begin
    H = Diagonal([1.0, 2.0, 3.0])

    function diagonal_fixture()
        initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
        initial[1, 1] = [reshape(ones(3), 3, 1)]
        return initial
    end

    function diagonal_A!(out, input)
        out[1, 1][1] .+= H * input[1, 1][1]
        return out
    end

    for alg in (:slow, :fast), position in 1 : 3
        initial = diagonal_fixture()
        val = SUNDMRG.Lanczos!(diagonal_A!, initial, position, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 3, alg = alg)
        @test val ≈ Float64(position) atol = 1e-12
        expected = zeros(3)
        expected[position] = 1.0
        @test abs.(vec(initial[1, 1][1])) ≈ expected atol = 1e-10
    end

    initial_eigenvector = diagonal_fixture()
    initial_eigenvector[1, 1][1] .= reshape([1.0, 0.0, 0.0], 3, 1)
    restarted_val = SUNDMRG.Lanczos!(diagonal_A!, initial_eigenvector, 2, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 3)
    @test restarted_val ≈ 2.0 atol = 1e-12

    @test_throws ArgumentError SUNDMRG.Lanczos!(diagonal_A!, diagonal_fixture(), 4, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 3)
    @test_throws ArgumentError SUNDMRG.Lanczos!(diagonal_A!, diagonal_fixture(), 4, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 4)
    @test_throws ArgumentError SUNDMRG.Lanczos!(diagonal_A!, diagonal_fixture(), 0, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 3)
    @test_throws ArgumentError SUNDMRG.Lanczos!(diagonal_A!, diagonal_fixture(), 1, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 0)
    @test_throws ArgumentError SUNDMRG.Lanczos!(diagonal_A!, diagonal_fixture(), 1, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 3, allow_fewer = :yes)

    initial = diagonal_fixture()
    val = SUNDMRG.Lanczos!(diagonal_A!, initial, 4, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 4, allow_fewer = true)
    @test val ≈ 3.0 atol = 1e-12
end

@testset "Lanczos reports nonconvergence" begin
    rng = MersenneTwister(17)
    matrix = randn(rng, 30, 30)
    H = Symmetric(2.0 .* matrix)
    initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
    initial[1, 1] = [reshape(randn(rng, 30), 30, 1)]

    function random_A!(out, input)
        out[1, 1][1] .+= H * input[1, 1][1]
        return out
    end

    @test_throws ErrorException SUNDMRG.Lanczos!(random_A!, initial, 1, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 1)
end

@testset "Node-local MPI context" begin
    local_rank, local_size = SUNDMRG._node_local_mpi_context(MPI.COMM_WORLD, MPI.Comm_rank(MPI.COMM_WORLD))
    @test 0 <= local_rank < local_size
    @test 1 <= local_size <= MPI.Comm_size(MPI.COMM_WORLD)
end
