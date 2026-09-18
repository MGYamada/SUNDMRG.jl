using LinearAlgebra
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "Lanczos multiplicity and initial eigendirections" begin
    function solve_diagonal_reference(levels, start, position, maxiter, alg; allow_fewer = false, seed = 1858)
        H = Diagonal(levels)
        initial = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
        initial[1, 1] = [reshape(copy(start), length(start), 1)]
        function diagonal_A!(out, input)
            out[1, 1][1] .+= H * input[1, 1][1]
            return out
        end
        Random.seed!(seed)
        energy = SUNDMRG.Lanczos!(
            diagonal_A!, initial, position, MPI.COMM_SELF, 0, CPUEngine;
            maxiter = maxiter, alg = alg, allow_fewer = allow_fewer,
        )
        return energy, vec(initial[1, 1][1])
    end

    fixtures = (
        (
            name = "identity needs only the requested two directions",
            levels = ones(4), start = ones(4), position = 2, maxiter = 2,
        ),
        (
            name = "isolated degenerate ground level before chain breakdown",
            levels = [-20.0; -20.0; collect(1.0 : 78.0)], start = ones(80),
            position = 2, maxiter = 80,
        ),
        (
            # One random direction plus this guess cannot span the two low
            # eigendirections: the guess has no overlap with either of them.
            name = "guess orthogonal to the degenerate low eigenspace",
            levels = [-20.0; -20.0; collect(1.0 : 78.0)],
            start = [zeros(2); 1.0; zeros(77)], position = 2, maxiter = 80,
        ),
        (
            # A useful excited solve must not require all 80 basis vectors just
            # because the supplied guess is already the exact ground state.
            name = "exact ground guess with a basis budget below full dimension",
            levels = [-20.0; collect(1.0 : 79.0)],
            start = [1.0; zeros(79)], position = 2, maxiter = 78,
        ),
    )

    for fixture in fixtures, alg in (:slow, :fast), seed in (1858, 2858, 3858)
        @testset "$(fixture.name), $alg, seed = $seed" begin
            energy, vector = solve_diagonal_reference(
                fixture.levels, fixture.start, fixture.position, fixture.maxiter, alg; seed = seed,
            )
            expected = fixture.levels[fixture.position]
            H = Diagonal(fixture.levels)
            @test energy ≈ expected atol = 1e-10 rtol = 0
            @test norm(vector) ≈ 1.0 atol = 1e-12 rtol = 0
            @test norm(H * vector - energy * vector) / max(norm(H), abs(energy), 1.0) <= 1e-8
            # Compare against the entire known eigenspace, including multiplicity.
            @test norm(vector[fixture.levels .!= expected]) <= 1e-8
        end
    end

    @testset "Unavailability differs from a basis budget" begin
        for alg in (:slow, :fast), sign in (-1.0, 1.0)
            @test_throws ArgumentError solve_diagonal_reference([3.0], [sign], 2, 2, alg)
            energy, vector = solve_diagonal_reference([3.0], [sign], 2, 2, alg; allow_fewer = true)
            @test energy ≈ 3.0 atol = 1e-12 rtol = 0
            @test norm(vector) ≈ 1.0 atol = 1e-12 rtol = 0
        end
    end
end
