@testset "run_DMRG regression" begin
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; correlation = :bad)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; alg = :bad)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; target = -1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; widthmax = -1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; margin = -1)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; tol_energy = 0.0)
    @test_throws ArgumentError run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; tol_EE = NaN)
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
