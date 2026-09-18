using LinearAlgebra
using MPI
using Random

MPI.Initialized() || MPI.Init()

@testset "Near-zero sweep convergence" begin
    tolerance = 1e-3
    # Negative energies and positive entropies use the same convergence rule.
    for sign in (-1.0, 1.0)
        previous = sign * 2e-12
        stable = previous + 1e-16
        unstable = sign * 1e-12
        @test SUNDMRG._relative_change_converged(stable, previous, tolerance)
        @test SUNDMRG._relative_change_converged(previous, stable, tolerance)
        @test !SUNDMRG._relative_change_converged(unstable, previous, tolerance)
        @test !SUNDMRG._relative_change_converged(previous, unstable, tolerance)
    end
    @test SUNDMRG._relative_change_converged(0.0, 0.0, tolerance)
    @test !SUNDMRG._relative_change_converged(-1e-12, 1e-12, tolerance)
    for invalid in (NaN, Inf, -Inf)
        @test !SUNDMRG._relative_change_converged(invalid, 0.0, tolerance)
        @test !SUNDMRG._relative_change_converged(0.0, invalid, tolerance)
    end

    config = SUNDMRG._FiniteRunConfig(
        Val(:square), 2, 2, 4, 2, (20, 0.0), Tuple{Int, Float64}[], (20, 0.0),
        0, 100, 0, nothing, Val(false), ".", 20.0, tolerance, tolerance, 1,
        Val(:none), 0, Val(:slow), false,
    )
    trivial = SUNDMRG.trivialirrep(Val(2))
    runtime = SUNDMRG._FiniteRuntime(
        CPUEngine, MPI.COMM_SELF, 0, 1, Val(true), true,
        typeof(trivial), [trivial], -1.0,
    )
    converged(energies, entropies) =
        SUNDMRG._update_measurement_flag(false, energies, entropies, config, runtime)

    stable_energies = [-2e-12, -2e-12 + 1e-16]
    stable_entropies = [2e-12, 2e-12 + 1e-16]
    @test converged(stable_energies, stable_entropies)
    @test converged([0.0, 0.0], [0.0, 0.0])
    @test !converged([-2e-12, -1e-12], stable_entropies)
    @test !converged(stable_energies, [0.0, 1e-12])
    @test !converged([-2e-12, NaN], stable_entropies)
    @test !converged(stable_energies, [2e-12, Inf])
    # One recorded sweep cannot establish convergence of either observable.
    @test !converged([-2e-12], [2e-12])
end

@testset "Density truncation across tied SU(2) irreps" begin
    # Spin 0, 1, and 2 sectors have dimensions 1, 3, and 5. The full density
    # operator has unit trace after repeating each eigenvalue by this dimension.
    betas = [SUNDMRG.SUNIrrep{2}((n, 0)) for n in (0, 2, 4)]
    dimensions = [1, 3, 5]
    eigenvalues = [[0.0, 2.0, 2.0], [1.0, 2.0], [2.0, 3.0]] ./ 38
    @test sum(dimensions .* sum.(eigenvalues)) ≈ 1.0 atol = 1e-15

    eigenvectors = [
        [1.0 -1.0 0.0; 1.0 1.0 0.0; 0.0 0.0 sqrt(2.0)] ./ sqrt(2.0),
        [3.0 -4.0; 4.0 3.0] ./ 5.0,
        [1.0 -1.0; 1.0 1.0] ./ sqrt(2.0),
    ]
    block_energies = [[10.0, 20.0, 30.0], [40.0, 50.0], [60.0, 70.0]]
    hamiltonians = [Q * Diagonal(E) * Q' for (Q, E) in zip(eigenvectors, block_energies)]
    block = (length = 2, scalar_dict = Dict(:H => hamiltonians))
    side = SUNDMRG._StepSideContext(3, [3, 2, 2], betas, nothing, 1, :l, nothing, block)
    context = SUNDMRG._StepDensityContext(MPI.COMM_SELF, 0, 1, CPUEngine, 0.0, 3, 20.0, false)

    entropy, spectrum, transforms, kept, projected, indices =
        SUNDMRG._density_truncation_basis(
            eigenvalues, eigenvectors, side, dimensions, block, block, context, Val(2), 1,
        )

    # Keep the largest value and exactly two of the four equal cutoff values.
    @test indices == [[false, true, true], [false, false], [false, true]]
    @test kept == [2, 0, 1]
    @test sum(kept) == 3
    @test sum(dimensions .* kept) == 7
    # The full spectrum contains 3 copies of 1/38, 10 of 2/38, and 5 of 3/38.
    expected_entropy = log(38) - (20log(2) + 15log(3)) / 38
    @test entropy ≈ expected_entropy atol = 1e-13
    @test all(all(isfinite, levels) for levels in values(spectrum))
    @test sort(length.(collect(values(spectrum)))) == [2, 2, 2]

    for sector in eachindex(kept)
        @test transforms[sector]' * transforms[sector] ≈
            Matrix{Float64}(I, kept[sector], kept[sector]) atol = 1e-14
        expected = Diagonal(block_energies[sector][indices[sector]])
        @test projected[sector] ≈ expected atol = 1e-12
    end
    for budget in 0 : 8
        keep = SUNDMRG._density_keep_indices(eigenvalues, budget)
        @test sum(count, keep) == min(budget, 7)
        retained = vcat((sector[mask] for (sector, mask) in zip(eigenvalues, keep))...)
        discarded = vcat((sector[.!mask] for (sector, mask) in zip(eigenvalues, keep))...)
        if !isempty(retained) && !isempty(discarded)
            @test minimum(retained) >= maximum(discarded)
        end
    end
end

@testset "Cooldown limit exhaustion" begin
    function cooldown_fixture(limit)
        Random.seed!(1858)
        run_DMRG(
            SU(2)HeisenbergModel(), SquareLattice(4, 2), 1, [1], 20, CPUEngine;
            max_cooldown_sweeps = limit, tol_energy = 1e-10, tol_EE = 1e-10,
            verbose = false, manage_mpi = false,
        )
    end

    # A deliberately truncated warmup and sweep must not be accepted as
    # converged after a single cooldown pass with the larger bond dimension.
    @test_throws r"DMRG did not converge within 1 cooldown sweeps" cooldown_fixture(1)
    rank, result = cooldown_fixture(8)
    @test rank == 0
    @test all(isfinite, result.energies)
    @test all(isfinite, result.EEs)
    @test 4 <= length(result.energies) <= 11 # Growth, one sweep, cooldown, measurement.
end
