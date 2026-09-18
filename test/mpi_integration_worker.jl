using Test
using SUNDMRG
using Random
using JLD2
using LinearAlgebra

include("reference_su2.jl")
include("mpi_collective_checks.jl")

const MPI = SUNDMRG.MPI
const output_dir = abspath(ARGS[1])
const expected_ranks = parse(Int, ARGS[2])

@testset "CPU MPI integration, $expected_ranks ranks" begin
    @test !MPI.Initialized()
    @test init_DMRG!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    @test MPI.Comm_size(comm) == expected_ranks
    MPICollectiveChecks.run_checks(comm)
    scratch = joinpath(output_dir, "scratch")
    if rank == 0
        mkpath(joinpath(scratch, "keepdir"))
        write(joinpath(scratch, "keep.txt"), "unrelated scratch data")
        write(joinpath(scratch, "keepdir", "keep.txt"), "unrelated nested data")
    end
    MPI.Barrier(comm)

    bonds = SU2DenseReference.square_cylinder_bonds(2, 4)
    reference = SU2DenseReference.singlet_reference(8, bonds)
    # The independent reference numbers each column upwards; DMRG reverses
    # every second column along its snake path. This is the only relabeling.
    snake_site = [1, 2, 3, 4, 8, 7, 6, 5]
    ground_state = reference.vectors[:, 1]
    correlations = Dict(
        minmax(snake_site[i], snake_site[j]) =>
        dot(ground_state, SU2DenseReference.heisenberg_matrix(8, [(i, j)]) * ground_state)
        for (i, j) in bonds
    )
    results = Dict{Tuple{Symbol, Bool, Int}, Any}()
    growth_result = nothing
    try
        for alg in (:slow, :fast), fileio in (false, true), target in (0, 1)
            @testset "$alg fileio=$fileio target=$target" begin
                Random.seed!(1858)
                returned_rank, result = run_DMRG(
                    SU(2)HeisenbergModel(), SquareLattice(2, 4), 128, [128], 128, CPUEngine;
                    alg = alg, target = target, fileio = fileio, scratch = scratch,
                    correlation = target == 0 ? :nn : :none,
                    max_cooldown_sweeps = 12, verbose = false, manage_mpi = false,
                )
                @test returned_rank == rank
                @test !MPI.Finalized()
                @test MPI.Allreduce(rank + 1, +, comm) == expected_ranks * (expected_ranks + 1) ÷ 2
                if rank == 0
                    @test result isa DMRGOutput
                    @test last(result.energies) ≈ reference.energies[target + 1] atol = 1e-10 rtol = 0
                    @test all(isfinite, result.energies)
                    @test all(isfinite, result.EEs)
                    @test maximum(abs, result.errors) <= 1e-12
                    if target == 0
                        @test Set(keys(result.SiSj)) == Set(keys(correlations))
                        @test all(isfinite, values(result.SiSj))
                        for bond in keys(correlations)
                            @test result.SiSj[bond] ≈ correlations[bond] atol = 1e-10 rtol = 0
                        end
                        @test sum(values(result.SiSj)) ≈ last(result.energies) atol = 1e-10 rtol = 0
                    else
                        @test isempty(result.SiSj)
                    end
                    @test sort(readdir(scratch)) == ["keep.txt", "keepdir"]
                    @test read(joinpath(scratch, "keep.txt"), String) == "unrelated scratch data"
                    @test read(joinpath(scratch, "keepdir", "keep.txt"), String) == "unrelated nested data"
                    results[(alg, fileio, target)] = (
                        energy = last(result.energies),
                        entropy = result.EE,
                        correlations = result.SiSj,
                    )
                else
                    @test result === nothing
                end
            end
        end
        @testset "4x2 growth with file storage" begin
            # Unlike 2x4, this geometry grows beyond the warmup lattice, which
            # exercises the distributed growth block/tensor storage paths.
            growth_reference = SU2DenseReference.singlet_reference(8, SU2DenseReference.square_cylinder_bonds(4, 2))
            Random.seed!(1858)
            returned_rank, result = run_DMRG(
                SU(2)HeisenbergModel(), SquareLattice(4, 2), 128, [128], 128, CPUEngine;
                alg = :fast, fileio = true, scratch = scratch,
                max_cooldown_sweeps = 12, verbose = false, manage_mpi = false,
            )
            @test returned_rank == rank
            @test !MPI.Finalized()
            @test MPI.Allreduce(rank + 1, +, comm) == expected_ranks * (expected_ranks + 1) ÷ 2
            if rank == 0
                @test result isa DMRGOutput
                @test last(result.energies) ≈ first(growth_reference.energies) atol = 1e-10 rtol = 0
                @test all(isfinite, result.energies)
                @test all(isfinite, result.EEs)
                @test maximum(abs, result.errors) <= 1e-12
                @test sort(readdir(scratch)) == ["keep.txt", "keepdir"]
                @test read(joinpath(scratch, "keep.txt"), String) == "unrelated scratch data"
                @test read(joinpath(scratch, "keepdir", "keep.txt"), String) == "unrelated nested data"
                growth_result = (energy = last(result.energies), entropy = result.EE)
            else
                @test result === nothing
            end
        end
        if rank == 0
            jldsave(joinpath(output_dir, "results.jld2"); results = results, growth_result = growth_result)
        end
    finally
        @test finalize_DMRG!()
    end
    @test MPI.Finalized()
    write(joinpath(output_dir, "rank-$rank.ok"), "rank=$rank size=$expected_ranks finalized=true\n")
end
println("MPI_INTEGRATION_OK ranks=$expected_ranks")
