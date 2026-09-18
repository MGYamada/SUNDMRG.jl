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

    abstract_eltype = Matrix{Vector}(undef, 1, 1)
    abstract_eltype[1, 1] = [reshape([1.0, 2.0], 2, 1)]
    @test SUNDMRG.mydot(abstract_eltype, abstract_eltype) == 5.0
    SUNDMRG.myzero!(abstract_eltype)
    @test iszero(only(only(abstract_eltype)))
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

@testset "Lanczos dense references for clustered and degenerate spectra" begin
    fixtures = (
        (name = "clustered", levels = [-2.0, -2.0 + 1e-5, -2.0 + 3e-5, -0.25, 0.1, 2.0, 4.0, 7.0], seed = 1858, unresolved_position = 7, insufficient_maxiter = 7),
        (name = "degenerate", levels = [-2.0, -2.0, -0.5, 0.5, 1.0, 1.0], seed = 2858, unresolved_position = 2, insufficient_maxiter = 4),
    )

    for fixture in fixtures
        @testset "$(fixture.name)" begin
            n = length(fixture.levels)
            rng = MersenneTwister(fixture.seed)
            rotation = Matrix(qr(randn(rng, n, n)).Q)
            H = Symmetric(rotation * Diagonal(fixture.levels) * rotation')
            reference = eigen(H)
            # Every eigendirection is represented in the starting vector. The
            # dense diagonalization supplies an independent ordering and basis.
            initial_data = rotation * collect(1.0 : n)

            function reference_A!(out, input)
                out[1, 1][1] .+= H * input[1, 1][1]
                return out
            end

            for alg in (:slow, :fast), position in 1 : n
                @testset "$alg, position = $position" begin
                    initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
                    initial[1, 1] = [reshape(copy(initial_data), n, 1)]
                    # Lanczos draws a random orthogonal restart at breakdown.
                    Random.seed!(fixture.seed + position)
                    value = SUNDMRG.Lanczos!(reference_A!, initial, position, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = n, alg = alg)
                    vector = vec(initial[1, 1][1])

                    @test value ≈ reference.values[position] atol = 1e-10 rtol = 0
                    @test norm(vector) ≈ 1.0 atol = 1e-12 rtol = 0
                    @test norm(H * vector - value * vector) / max(norm(H), abs(value), 1.0) <= 1e-8

                    multiplicity = findall(==(fixture.levels[position]), fixture.levels)
                    if length(multiplicity) > 1
                        # Degenerate eigenvectors may rotate or change sign.
                        # Compare their projections into the whole invariant
                        # subspace, not individual dense eigenvectors.
                        subspace = reference.vectors[:, multiplicity]
                        @test norm(vector - subspace * (subspace' * vector)) <= 1e-8
                    end
                end
            end

            for alg in (:slow, :fast)
                initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
                initial[1, 1] = [reshape(copy(initial_data), n, 1)]
                Random.seed!(fixture.seed)
                # A small residual alone must not accept the wrong index when
                # the basis budget cannot resolve lower levels or multiplicity.
                @test_throws ErrorException SUNDMRG.Lanczos!(reference_A!, initial, fixture.unresolved_position, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = fixture.insufficient_maxiter, alg = alg)
            end
        end
    end
end

@testset "Lanczos preserves a converged guess in a degenerate target space" begin
    rng = MersenneTwister(4858)
    rotation = Matrix(qr(randn(rng, 6, 6)).Q)
    H = Symmetric(rotation * Diagonal([-3.0, -2.0, -2.0, 0.5, 1.0, 3.0]) * rotation')
    guess = normalize(rotation[:, 2] + 2.0 .* rotation[:, 3])

    function degenerate_A!(out, input)
        out[1, 1][1] .+= H * input[1, 1][1]
        return out
    end

    for alg in (:slow, :fast), seed in (1858, 2858, 3858)
        initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
        initial[1, 1] = [reshape(copy(guess), 6, 1)]
        Random.seed!(seed)
        value = SUNDMRG.Lanczos!(degenerate_A!, initial, 2, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 6, alg = alg)
        vector = vec(initial[1, 1][1])

        @test value ≈ -2.0 atol = 1e-10 rtol = 0
        @test norm(vector) ≈ 1.0 atol = 1e-12 rtol = 0
        @test norm(H * vector - value * vector) / max(norm(H), abs(value), 1.0) <= 1e-8
        # Selecting a new arbitrary direction at every solve changes observables
        # such as entanglement despite an unchanged degenerate target energy.
        @test abs(dot(vector, guess)) ≈ 1.0 atol = 1e-12 rtol = 0
    end
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

    for alg in (:slow, :fast)
        @test_throws ErrorException SUNDMRG.Lanczos!(random_A!, deepcopy(initial), 1, MPI.COMM_SELF, 0, SUNDMRG.CPUEngine; maxiter = 1, alg = alg)
    end
end

@testset "Node-local MPI context" begin
    local_rank, local_size = SUNDMRG._node_local_mpi_context(MPI.COMM_WORLD, MPI.Comm_rank(MPI.COMM_WORLD))
    @test 0 <= local_rank < local_size
    @test 1 <= local_size <= MPI.Comm_size(MPI.COMM_WORLD)
end
