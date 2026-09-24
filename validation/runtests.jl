using Test
using LinearAlgebra
using Random

include("sun_ed.jl")
include("sun_ed_krylov.jl")
include(joinpath(@__DIR__, "..", "test", "reference_sun.jl"))

const ED = SUNExactDiagonalization
const EK = SUNEDKrylov

@testset "Independent block Krylov ED" begin
    @testset "Guard levels and independent color singlets" begin
        sector = ED.SYTSector(3, [2, 2, 2])
        bonds = SUNColorReference.square_cylinder_bonds(2, 3)
        H = ED.PermutationHamiltonian(sector, bonds)
        reference = SUNColorReference.singlet_reference(3, 6, bonds)
        result = EK.krylov_spectrum(H; levels = 3)
        # The third level belongs to an exact two-fold cluster: include both.
        @test length(result.energies) == 4
        @test result.clusters == [1:1, 2:2, 3:4]
        @test result.energies ≈ reference.energies[1:4] atol = 1e-10
        @test result.guard_energies ≈ reference.energies[5:5] atol = 1e-10
        @test result.max_checked_residual < 1e-12
        @test result.solve_matvecs > 0
        @test result.verification_matvecs == sector.dimension
        @test result.solve_seconds >= 0 && result.verification_seconds >= 0
    end

    @testset "Restarted solve, weighted bonds and seed independence" begin
        sector = ED.SYTSector(2, [5, 5]) # dimension 42
        bonds = [(i, i + 1, 1 + i / 19) for i in 1:9]
        append!(bonds, [(1, 10, 0.43), (2, 8, -0.17), (10, 1, 0.21)])
        H = ED.PermutationHamiltonian(sector, bonds)
        reference = ED.dense_spectrum(H)
        for seed in (1584, 1585)
            result = EK.krylov_spectrum(H; levels = 3, blocksize = 4,
                                        krylovdim = 20, seed)
            @test result.energies ≈ reference.energies[1:3] atol = 1e-10
            @test result.max_checked_residual < 1e-12
            @test result.orthogonality < 1e-10
            @test result.iterations > 1
            @test result.solve_matvecs > 20
            for k in 1:3
                @test abs(dot(result.vectors[k], reference.vectors[:, k])) ≈ 1 atol = 1e-10
            end
        end
        @test_throws ErrorException EK.krylov_spectrum(H; levels = 3, blocksize = 4,
                                                       krylovdim = 8, maxiter = 1)
    end

    @testset "Degenerate projectors and insufficient blocks" begin
        sector = ED.SYTSector(4, [2, 2, 2, 2]) # dimension 14
        H = ED.PermutationHamiltonian(sector, [(1, 2)])
        # P^2=I fixes the negative eigenspace projector independently of a solver.
        projector = (I - ED.dense_matrix(H)) / 2
        @test tr(projector) ≈ 9 atol = 1e-12
        for seed in (1584, 1585)
            result = EK.krylov_spectrum(H; levels = 2, blocksize = 10, seed)
            @test result.energies ≈ fill(-1.0, 9) atol = 1e-12
            V = hcat(result.vectors...)
            @test V * V' ≈ projector atol = 1e-10
            @test result.clusters == [1:9]
        end
        @test_throws ErrorException EK.krylov_spectrum(H; levels = 2, blocksize = 4)
        zero = ED.PermutationHamiltonian(sector, [])
        @test_throws ErrorException EK.krylov_spectrum(zero; levels = 1, blocksize = 4)
        full = EK.krylov_spectrum(zero; levels = 1, blocksize = 14)
        @test full.energies == zeros(14)
        @test full.clusters == [1:14]
        single = ED.PermutationHamiltonian(ED.SYTSector(3, [1]), [])
        @test EK.krylov_spectrum(single; levels = 1).energies == [0.0]
    end

    @testset "Input and preallocation guards" begin
        @test_throws ArgumentError ED.SYTSector(3, [4, 4, 4]; max_cache_bytes = 1)
        @test_throws ArgumentError ED.sector_spec(3, [1]; max_cache_bytes = 0)
        spec = ED.sector_spec(4, [4, 4, 4, 4])
        @test spec.dimension == 24024
        @test spec.cache_bytes == 24024 * 15 * (2sizeof(Float64) + sizeof(Int))
        memory = EK.memory_estimate(24024, 16; blocksize = 8, krylovdim = 64)
        @test memory.working < memory.dense_matrix
        @test EK.memory_estimate(typemax(Int), 16; blocksize = 8, krylovdim = 64).dense_matrix > typemax(Int)
        H = ED.PermutationHamiltonian(ED.SYTSector(3, [2, 2, 2]), [(1, 2)])
        for kw in ((; levels = 0), (; levels = 6), (; blocksize = 0), (; blocksize = 2),
                   (; krylovdim = 4), (; maxiter = 0), (; tol = NaN), (; tol = 0),
                   (; cluster_tol = 1e-14), (; max_memory_bytes = 1), (; seed = -1))
            @test_throws ArgumentError EK.krylov_spectrum(H; levels = 3, kw...)
        end
    end
end
