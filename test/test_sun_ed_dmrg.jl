using JLD2

# Included inside the ordinary DMRG test's caller-owned MPI lifetime.
@testset "SU(3) DMRG against SYT and color-product ED" begin
    nc, Lx, Ly = 3, 2, 3
    nsites = Lx * Ly
    bonds = SUNColorReference.square_cylinder_bonds(Lx, Ly)
    reference = SUNColorReference.singlet_reference(nc, nsites, bonds)
    sector = ED.SYTSector(nc, [2, 2, 2])
    H = ED.PermutationHamiltonian(sector, bonds)
    spectrum = ED.dense_spectrum(H)
    @test spectrum.energies ≈ reference.energies atol = 1e-12 rtol = 0

    tables = JLD2.load(joinpath(@__DIR__, "..", "jld2", "table_SU3_13.jld2"), "tables")
    # Every proper block has <= 5 sites. Width 6 and 3^5 multiplets exceed
    # all possible row lengths and even the full product-space dimension.
    widthmax, m = 6, nc^(nsites - 1)
    expected_correlations = Dict{Tuple{Int, Int}, Float64}()
    ψ = reference.vectors[:, 1]
    for (i, j) in bonds
        P = SUNColorReference.hamiltonian(reference.states, [(i, j)])
        key = minmax(SUNColorReference.snake_site(Ly, i), SUNColorReference.snake_site(Ly, j))
        expected_correlations[key] = ED.dmrg_correlation(nc, dot(ψ, P * ψ))
    end

    for alg in (:slow, :fast), fileio in (false, true), target in (0, 1)
        @testset "$alg fileio=$fileio target=$target" begin
            mktempdir() do scratch
                Random.seed!(1583)
                rank, output = run_DMRG(SU(nc)HeisenbergModel(), SquareLattice(Lx, Ly),
                    m, [m], m, CPUEngine; widthmax = widthmax, tables = tables,
                    target = target, alg = alg, fileio = fileio, scratch = scratch,
                    correlation = target == 0 ? :nn : :none,
                    max_cooldown_sweeps = 12, verbose = false, manage_mpi = false)
                @test rank == 0
                if rank == 0
                    # Reported SU(3) energy uses sum(P), whereas SiSj uses P-1/3.
                    @test last(output.energies) ≈ reference.energies[target + 1] atol = 1e-10 rtol = 0
                    @test maximum(abs, output.errors) < 1e-12
                    if target == 0
                        actual = Dict(minmax(pair...) => value for (pair, value) in output.SiSj)
                        @test keys(actual) == keys(expected_correlations)
                        for pair in keys(expected_correlations)
                            @test actual[pair] ≈ expected_correlations[pair] atol = 1e-10 rtol = 0
                        end
                        @test sum(values(actual)) + length(bonds) / nc ≈ last(output.energies) atol = 1e-10
                    end
                end
                @test isempty(readdir(scratch))
            end
        end
    end
end
