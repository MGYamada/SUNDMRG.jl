@testset "run_DMRG regression" begin
    init_DMRG!()
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

        rank_nn, dmrg_nn = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(2, 2), 20, [20], 20, CPUEngine; correlation = :nn, verbose = false, manage_mpi = false)

        @test rank_nn == 0
        if rank_nn == 0
            @test length(dmrg_nn.energies) == 4
            @test last(dmrg_nn.energies) ≈ -3.2320508075688767 atol = 1e-12
            @test sort(collect(keys(dmrg_nn.SiSj))) == [(1, 2), (1, 4), (2, 3), (3, 4)]
        end
    finally
        finalize_DMRG!()
    end
end
