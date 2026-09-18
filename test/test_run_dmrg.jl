using LinearAlgebra
using Random

include("reference_su2.jl")

@testset "run_DMRG regression" begin
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; correlation = :bad)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; alg = :bad)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; target = -1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; lanczos_maxiter = 0)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; lanczos_maxiter = 1.5)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; target = 1, lanczos_maxiter = 1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; widthmax = -1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; margin = -1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; tol_energy = 0.0)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; tol_EE = NaN)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; max_cooldown_sweeps = 0)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; max_cooldown_sweeps = 1.5)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; fileio = :yes)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; verbose = :yes)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; manage_mpi = :yes)
    @test_throws ArgumentError run_DMRG(SU(3)HeisenbergModel(), SquareLattice(4, 3), 20, [20], 20, CPUEngine; widthmax = 0, tables = ())
    @test_throws ArgumentError run_DMRG(SU(3)HeisenbergModel(), SquareLattice(4, 3), 20, [20], 20, CPUEngine; widthmax = 3, tables = nothing)
    @test_throws ArgumentError SUNDMRG._init_runtime_and_engine(CPUEngine, :square, 3, 4, 2, 0, 1)

    did_init = init_DMRG!()
    @test did_init isa Bool
    @test SUNDMRG.MPI.Initialized()
    @test !SUNDMRG.MPI.Finalized()
    @test init_DMRG!() == false
    try
        @testset "independent SU(2) singlet energies" begin
            # A single physical bond fixes the spin normalization: singlet -3/4,
            # triplet +1/4. No package Hamiltonian or coefficient table is used.
            dimer = SU2DenseReference.singlet_reference(2, [(1, 2)])
            @test dimer.energies ≈ [-0.75] atol = 1e-14
            @test eigvals(Symmetric(dimer.H)) ≈ [-0.75, 0.25] atol = 1e-14

            for (Lx, Ly) in ((2, 2), (2, 4))
                @testset "$(Lx)x$(Ly) cylinder" begin
                    nsites = Lx * Ly
                    bonds = SU2DenseReference.square_cylinder_bonds(Lx, Ly)
                    @test length(bonds) == Lx * Ly + (Lx - 1) * Ly
                    unordered_bonds = [minmax(i, j) for (i, j) in bonds]
                    @test length(unique(unordered_bonds)) == (Ly == 2 ? Lx + (Lx - 1) * Ly : length(bonds))

                    reference = SU2DenseReference.singlet_reference(nsites, bonds)
                    nsinglets = binomial(nsites, nsites ÷ 2) - binomial(nsites, nsites ÷ 2 - 1)
                    @test length(reference.energies) == nsinglets
                    @test reference.S2 * reference.vectors ≈ zero(reference.vectors) atol = 1e-12
                    @test reference.H * reference.vectors ≈ reference.vectors * Diagonal(reference.energies) atol = 1e-12

                    # run_DMRG targets the global SU(2) singlet sector here.
                    # For 2x2, target = 1 is NOT the lowest triplet; explicitly
                    # guard against accidentally using an Sz-only reference.
                    if (Lx, Ly) == (2, 2)
                        @test reference.energies ≈ [-1.5 - sqrt(3), -1.5 + sqrt(3)] atol = 1e-12
                        @test reference.energies[2] > eigvals(Symmetric(reference.H))[2] + 1
                    else
                        # The first excited singlet is twofold degenerate. Its
                        # entropy depends on the direction within this subspace,
                        # so preserving an already-converged prediction matters.
                        first_excited = -1 - sqrt(2)
                        @test reference.energies[2 : 3] ≈ fill(first_excited, 2) atol = 1e-12 rtol = 0
                        @test count(E -> abs(E - first_excited) <= 1e-12, reference.energies) == 2
                    end

                    # This is at least the full product-space dimension of
                    # every proper block, so the fixture has no DMRG truncation.
                    # atol=1e-10 leaves room for Float64 Lanczos/recoupling roundoff
                    # on these <=8-site systems, with no truncation error budget.
                    m = 1 << (nsites - 1)
                    for alg in (:slow, :fast), target in (0, 1)
                        @testset "$alg target=$target" begin
                            # A bounded, reproducible run catches arbitrary
                            # rotations of degenerate states between sweeps,
                            # which otherwise keep EE from converging.
                            Random.seed!(1858)
                            rank_ref, dmrg_ref = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(Lx, Ly), m, [m], m, CPUEngine; target = target, alg = alg, max_cooldown_sweeps = 12, verbose = false, manage_mpi = false)
                            @test rank_ref == 0
                            if rank_ref == 0
                                @test last(dmrg_ref.energies) ≈ reference.energies[target + 1] atol = 1e-10 rtol = 0
                                @test all(isfinite, dmrg_ref.energies)
                                @test maximum(abs, dmrg_ref.errors) <= 1e-12
                            end
                        end
                    end
                end
            end
        end

        rank, dmrg = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(4, 4), 100, [100, 200, 400, 800], 1600, CPUEngine; verbose = false, manage_mpi = false)

        @test rank == 0
        if rank == 0
            @test length(dmrg.energies) == 7
            @test last(dmrg.energies) ≈ -10.264289620978833 atol = 1e-10
            @test length(dmrg.errors) == length(dmrg.energies)
            @test length(dmrg.EEs) == length(dmrg.energies)
        end

        mktempdir() do scratch
            rank_fileio, dmrg_fileio = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; fileio = true, scratch = scratch, verbose = false, manage_mpi = false)

            @test rank_fileio == 0
            if rank_fileio == 0
                @test length(dmrg_fileio.energies) == 4
                @test last(dmrg_fileio.energies) ≈ -3.2320508075688767 atol = 1e-12
                @test isempty(readdir(scratch))
            end
        end

        rank_fast, dmrg_fast = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; alg = :fast, verbose = false, manage_mpi = false)

        @test rank_fast == 0
        if rank_fast == 0
            @test length(dmrg_fast.energies) == 4
            @test last(dmrg_fast.energies) ≈ -3.2320508075688767 atol = 1e-12
            @test isempty(dmrg_fast.SiSj)
        end

        rank_excited, dmrg_excited = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; target = 1, lanczos_maxiter = 2, verbose = false, manage_mpi = false)

        @test rank_excited == 0
        if rank_excited == 0
            @test length(dmrg_excited.energies) == 4
            @test last(dmrg_excited.energies) ≈ 0.2320508075688773 atol = 1e-12
            @test last(dmrg_excited.energies) > last(dmrg_fast.energies)
        end

        @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; target = 2, verbose = false, manage_mpi = false)

        rank_nn, dmrg_nn = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; correlation = :nn, verbose = false, manage_mpi = false)

        @test rank_nn == 0
        if rank_nn == 0
            @test length(dmrg_nn.energies) == 4
            @test last(dmrg_nn.energies) ≈ -3.2320508075688767 atol = 1e-12
            @test sort(collect(keys(dmrg_nn.SiSj))) == [(1, 2), (1, 4), (2, 3), (3, 4)]
        end

        rank_chain, dmrg_chain = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; correlation = :chain, margin = 0, verbose = false, manage_mpi = false)

        @test rank_chain == 0
        if rank_chain == 0
            @test length(dmrg_chain.energies) == 4
            @test last(dmrg_chain.energies) ≈ -3.2320508075688767 atol = 1e-12
            @test sort(collect(keys(dmrg_chain.SiSj))) == [(1, 4)]
            @test dmrg_chain.SiSj[(1, 4)] ≈ -0.25 atol = 1e-12
        end
    finally
        @test finalize_DMRG!()
        @test finalize_DMRG!() == false
    end
end
