using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "validation", "sun_ed.jl"))
include("reference_sun.jl")

const ED = SUNExactDiagonalization

function ed_partitions(n, max_rows; largest = n)
    n == 0 && return [Int[]]
    max_rows == 0 && return Vector{Int}[]
    [vcat(first, rest) for first in min(n, largest) : -1 : 1
                     for rest in ed_partitions(n - first, max_rows - 1; largest = first)]
end

function ed_irrep_dimension(nc, shape)
    rows = vcat(shape, zeros(Int, nc - length(shape)))
    product = big(1) // big(1)
    for i in 1 : nc, j in i + 1 : nc
        product *= (rows[i] - rows[j] + j - i) // (j - i)
    end
    @assert denominator(product) == 1
    Int(numerator(product))
end

@testset "SU(N) SYT exact diagonalization" begin
    @testset "Sector validation and size guards" begin
        for (nc, shape) in ((1, [1]), (true, [1]), (3, Int[]), (2, [1, 1, 1]),
                            (3, [1, 2]), (3, [2, 0]), (3, [2.0, 1.0]))
            @test_throws ArgumentError ED.SYTSector(nc, shape)
        end
        @test_throws ArgumentError ED.SYTSector(4, [4, 4, 4, 4]; max_dimension = 100)
        @test_throws ArgumentError ED.SYTSector(3, [2, 2, 2]; max_dimension = 0)
        @test_throws ArgumentError ED.SYTSector(3, [4, 4, 4]; max_cache_bytes = 1)
        @test_throws ArgumentError ED.sector_spec(3, [1]; max_cache_bytes = 0)
        spec = ED.sector_spec(4, [4, 4, 4, 4])
        @test spec.dimension == 24024
        @test spec.cache_bytes == 24024 * 15 * (2sizeof(Float64) + sizeof(Int))
        for (nc, shape, d) in ((3, [2, 2, 2], 5), (3, [4, 4, 4], 462),
                               (4, [2, 2, 2, 2], 14))
            @test ED.SYTSector(nc, shape).dimension == d
        end
        single = ED.PermutationHamiltonian(ED.SYTSector(3, [1]), [])
        @test ED.dense_spectrum(single).energies == [0.0]
    end

    @testset "Symmetric-group relations and central element" begin
        # Exercise all shapes, including one-row, one-column, and mixed shapes.
        for shape in ed_partitions(5, 5)
            sector = ED.SYTSector(5, shape)
            matrices = [ED.dense_matrix(ED.PermutationHamiltonian(sector, [(k, k + 1)])) for k in 1 : 4]
            for S in matrices
                @test S ≈ S' atol = 1e-14
                @test S * S ≈ I atol = 1e-14
            end
            for k in 1 : 3
                A, B = matrices[k], matrices[k + 1]
                @test A * B * A ≈ B * A * B atol = 1e-14
            end
            for k in 1 : 4, l in k + 2 : 4
                @test matrices[k] * matrices[l] ≈ matrices[l] * matrices[k] atol = 1e-14
            end
            content_sum = sum(c - r for r in eachindex(shape) for c in 1 : shape[r])
            complete = ED.PermutationHamiltonian(sector, [(i, j) for i in 1 : 5 for j in i + 1 : 5])
            @test ED.dense_matrix(complete) ≈ content_sum * I atol = 1e-13
        end
    end

    @testset "All-sector spectra match independent color swaps" begin
        for (nc, nsites) in ((2, 4), (3, 3), (4, 4))
            bonds = [(i, j, (-1.0)^(i + j) * (2i + j) / 7)
                     for i in 1 : nsites for j in i + 1 : nsites]
            # A duplicate with reversed endpoints must add, not be deduplicated.
            push!(bonds, (nsites, 1, 0.125))
            expected = eigvals(Symmetric(SUNColorReference.hamiltonian(
                SUNColorReference.color_basis(nc, nsites), bonds)))
            actual = Float64[]
            for shape in ed_partitions(nsites, nc)
                H = ED.PermutationHamiltonian(ED.SYTSector(nc, shape), bonds)
                result = ED.dense_spectrum(H)
                @test maximum(result.residuals) < 1e-13
                append!(actual, repeat(result.energies; inner = ed_irrep_dimension(nc, shape)))
            end
            @test length(actual) == nc^nsites
            @test sort(actual) ≈ expected atol = 1e-12 rtol = 0
        end
    end

    @testset "Dense-vector action remains linear below sparse pruning tolerance" begin
        sector = ED.SYTSector(3, [2, 2, 2])
        H = ED.PermutationHamiltonian(sector, [(1, 6, 0.7), (2, 4, -1.2), (6, 1, 0.3)])
        matrix = ED.dense_matrix(H)
        source = randn(MersenneTwister(158), sector.dimension)
        dest = similar(source)
        workspace = ED.EDWorkspace(H)
        ED.apply!(dest, H, source, workspace)
        @test dest ≈ matrix * source atol = 1e-14
        original = copy(source)
        ED.apply!(dest, H, 1e-16 * source, workspace)
        @test dest ≈ 1e-16 * (matrix * source) atol = 1e-29 rtol = 1e-13
        @test source == original
        @test_throws ArgumentError ED.apply!(source, H, source, workspace)
        @test_throws ArgumentError ED.apply!(dest, H, source, ED.EDWorkspace(dest, similar(dest)))
        @test_throws DimensionMismatch ED.apply!(zeros(1), H, source)
        @test_throws ArgumentError ED.dense_matrix(H; max_dimension = 4)
        for bonds in ([(0, 1)], [(1, 1)], [(1, 7)], [(1, 2, Inf)], [(1, 2, NaN)], [(1, 2, 1, 2)])
            @test_throws ArgumentError ED.PermutationHamiltonian(sector, bonds)
        end
        @test_throws ArgumentError ED.permutation_expectation(sector, zeros(5), 1, 2)
    end

    @testset "Singlet sectors and physical normalization" begin
        for nc in 2 : 4
            sector = ED.SYTSector(nc, [1, 1])
            H = ED.PermutationHamiltonian(sector, [(1, 2)])
            @test ED.dense_spectrum(H).energies == [-1.0]
            @test ED.dmrg_energy(H, -1.0) == (nc == 2 ? -0.75 : -1.0)
            @test ED.dmrg_correlation(nc, -1.0) ≈ (nc == 2 ? -0.75 : -1 - 1/nc)
        end
        for (nc, Lx, Ly) in ((2, 2, 2), (3, 2, 3), (4, 2, 2))
            nsites = Lx * Ly
            bonds = SUNColorReference.square_cylinder_bonds(Lx, Ly)
            ref = SUNColorReference.singlet_reference(nc, nsites, bonds)
            sector = ED.SYTSector(nc, fill(nsites ÷ nc, nc))
            H = ED.PermutationHamiltonian(sector, bonds)
            result = ED.dense_spectrum(H)
            @test size(ref.singlets, 2) == sector.dimension
            @test norm(ref.C2 * ref.vectors) < 1e-11
            @test result.energies ≈ ref.energies atol = 1e-12 rtol = 0
            @test maximum(result.residuals) < 1e-13
            @test all(result.energies[2:end] .> result.energies[1] + 1e-8)
            ψ, ψref = result.vectors[:, 1], ref.vectors[:, 1]
            for i in 1 : nsites, j in i + 1 : nsites
                P = SUNColorReference.hamiltonian(ref.states, [(i, j)])
                @test ED.permutation_expectation(sector, ψ, i, j) ≈ dot(ψref, P * ψref) atol = 1e-12
            end
            @test sum(ED.permutation_expectation(sector, ψ, i, j) for (i, j) in bonds) ≈ result.energies[1] atol = 1e-12
        end
    end
end
