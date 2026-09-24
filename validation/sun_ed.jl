module SUNExactDiagonalization

using LinearAlgebra
using SUNDMRG: RepresentationTheory

const RT = RepresentationTheory

# These transitions are extracted from the existing SYT routines, not from
# DMRG's coefficient tables. Each generator has at most two entries per column.
struct AdjacentTransposition
    diagonal::Vector{Float64}
    partner::Vector{Int}
    offdiagonal::Vector{Float64}
end

struct SYTSector
    nc::Int
    shape::Vector{Int}
    nsites::Int
    dimension::Int
    generators::Vector{AdjacentTransposition}
end

function positive_int(value, name)
    value isa Integer && !(value isa Bool) && 0 < value <= typemax(Int) ||
        throw(ArgumentError("$name must be a positive machine integer"))
    Int(value)
end

"""Validate a sector and estimate transition-array payload before building its graph."""
function sector_spec(nc, shape; max_dimension = 100_000, max_cache_bytes = 512 * 2^20)
    nc = positive_int(nc, "nc")
    nc >= 2 || throw(ArgumentError("nc must be at least two"))
    limit = positive_int(max_dimension, "max_dimension")
    cache_limit = positive_int(max_cache_bytes, "max_cache_bytes")
    rows = [positive_int(row, "row length") for row in shape]
    !isempty(rows) && length(rows) <= nc && issorted(rows; rev = true) ||
        throw(ArgumentError("shape must be a nonempty partition with at most nc rows"))
    nsites = positive_int(sum(big, rows), "site count")
    count = RT.multiplicity(rows)
    0 < count <= limit || throw(ArgumentError("SYT dimension $count exceeds max_dimension=$limit"))
    dimension = Int(count)
    cache_bytes = big(dimension) * (nsites - 1) * (2sizeof(Float64) + sizeof(Int))
    cache_bytes <= cache_limit || throw(ArgumentError("transition payload $cache_bytes exceeds max_cache_bytes=$cache_limit"))
    (; nc, shape = rows, nsites, dimension, cache_bytes = Int(cache_bytes))
end

"""
    SYTSector(nc, shape; max_dimension = 100_000, max_cache_bytes = 512 * 2^20)

Construct the complete multiplicity space for a Young diagram with positive,
nonincreasing row lengths and at most `nc` rows. Keep all `L = sum(shape)` boxes,
including full columns of height `nc`. The singlet shape is `fill(L ÷ nc, nc)`.
Dimension and transition-payload limits are checked before graph construction.
The cache limit excludes graph temporaries, array headers, and allocator overhead.
"""
function SYTSector(nc, shape; max_dimension = 100_000, max_cache_bytes = 512 * 2^20)
    spec = sector_spec(nc, shape; max_dimension, max_cache_bytes)
    nc, rows, nsites, dimension = spec.nc, spec.shape, spec.nsites, spec.dimension
    V, E, D = RT.SYTdiagram(rows)
    B, F = RT.bf(nsites, V, E, Int)
    B[1][1] == dimension || error("SYT graph and hook-length dimensions disagree")

    generators = AdjacentTransposition[]
    for k in 1 : nsites - 1
        diagonal = zeros(dimension)
        partner = collect(1 : dimension)
        offdiagonal = zeros(dimension)
        for column in 1 : dimension
            unit = RT.sparsevec2([column], [1.0], dimension)
            image = RT.Papply2!(unit, k, rows, nsites, E, D, F, Int)
            for (row, coefficient) in zip(image.nzind, image.nzval)
                if row == column
                    diagonal[column] = coefficient
                else
                    partner[column] = row
                    offdiagonal[column] = coefficient
                end
            end
        end
        push!(generators, AdjacentTransposition(diagonal, partner, offdiagonal))
    end
    SYTSector(nc, rows, nsites, dimension, generators)
end

struct PermutationHamiltonian
    sector::SYTSector
    bonds::Vector{Tuple{Int, Int, Float64}}
    words::Vector{Vector{Int}}
end

"""
    PermutationHamiltonian(sector, bonds)

Represent `sum(J * P(i,j))`. Each bond is `(i,j)` or `(i,j,J)` with a real,
finite coupling. Duplicate bonds are retained. Self-bonds are rejected.
"""
function PermutationHamiltonian(sector::SYTSector, bonds)
    weighted = Tuple{Int, Int, Float64}[]
    words = Vector{Int}[]
    for bond in bonds
        length(bond) in (2, 3) || throw(ArgumentError("a bond must have two or three entries"))
        i, j = minmax(positive_int(bond[1], "site"), positive_int(bond[2], "site"))
        i < j <= sector.nsites || throw(ArgumentError("bond sites must be distinct and within the lattice"))
        J = length(bond) == 2 ? 1.0 : bond[3]
        J isa Real && !(J isa Bool) && isfinite(J) || throw(ArgumentError("coupling must be finite and real"))
        J = Float64(J)
        isfinite(J) || throw(ArgumentError("coupling must fit in Float64"))
        push!(weighted, (i, j, J))
        # The palindrome acts in either reading direction. Never assume an
        # arbitrary transposition has only two nonzero entries per column.
        push!(words, vcat(collect(i : j - 1), collect(j - 2 : -1 : i)))
    end
    PermutationHamiltonian(sector, weighted, words)
end

struct EDWorkspace
    first::Vector{Float64}
    second::Vector{Float64}
end

EDWorkspace(H::PermutationHamiltonian) = EDWorkspace(zeros(H.sector.dimension), zeros(H.sector.dimension))

function apply_adjacent!(dest, generator, source)
    @inbounds for column in eachindex(source)
        dest[column] = generator.diagonal[column] * source[column]
    end
    @inbounds for column in eachindex(source)
        dest[generator.partner[column]] += generator.offdiagonal[column] * source[column]
    end
    dest
end

"""
    apply!(dest, H, source, workspace = EDWorkspace(H))

Apply the permutation Hamiltonian to a Float64 vector without pruning amplitudes
or materializing H. All four buffers must be disjoint. Workspaces can be reused
between calls, but must not be shared by concurrent calls.
"""
function apply!(dest::AbstractVector{Float64}, H::PermutationHamiltonian,
                source::AbstractVector{Float64}, workspace::EDWorkspace = EDWorkspace(H))
    buffers = (dest, source, workspace.first, workspace.second)
    for buffer in buffers
        axes(buffer) == (Base.OneTo(H.sector.dimension),) || throw(DimensionMismatch("ED vector size mismatch"))
    end
    for i in 1 : 3, j in i + 1 : 4
        Base.mightalias(buffers[i], buffers[j]) && throw(ArgumentError("ED buffers must not alias"))
    end
    fill!(dest, 0.0)
    for ((_, _, J), word) in zip(H.bonds, H.words)
        copyto!(workspace.first, source)
        current, next = workspace.first, workspace.second
        for k in word
            apply_adjacent!(next, H.sector.generators[k], current)
            current, next = next, current
        end
        @. dest += J * current
    end
    dest
end

"""
    dense_matrix(H; max_dimension = 2048)

Materialize a small reference Hamiltonian. Larger sectors use the opt-in block
Krylov driver; the size guard prevents accidental quadratic allocation.
"""
function dense_matrix(H::PermutationHamiltonian; max_dimension = 2048)
    limit = positive_int(max_dimension, "max_dimension")
    d = H.sector.dimension
    d <= limit || throw(ArgumentError("dense dimension $d exceeds max_dimension=$limit"))
    matrix = zeros(d, d)
    unit = zeros(d)
    workspace = EDWorkspace(H)
    for column in 1 : d
        fill!(unit, 0.0)
        unit[column] = 1.0
        apply!(view(matrix, :, column), H, unit, workspace)
    end
    isapprox(matrix, matrix'; atol = 64eps(Float64) * max(1, sum(abs(b[3]) for b in H.bonds; init = 0.0)),
             rtol = 64eps(Float64)) || error("permutation Hamiltonian is not symmetric")
    matrix
end

"""
    dense_spectrum(H; max_dimension = 2048)

Return all sector energies (including repeated levels), vectors, and normalized
residuals. Energies use the permutation convention, not a centered bond operator.
"""
function dense_spectrum(H::PermutationHamiltonian; max_dimension = 2048)
    spectrum = eigen(Symmetric(dense_matrix(H; max_dimension = max_dimension)))
    workspace = EDWorkspace(H)
    residual = zeros(H.sector.dimension)
    bound = sum(abs(b[3]) for b in H.bonds; init = 0.0)
    residuals = map(eachindex(spectrum.values)) do k
        ψ = view(spectrum.vectors, :, k)
        apply!(residual, H, ψ, workspace)
        @. residual -= spectrum.values[k] * ψ
        norm(residual) / max(bound, abs(spectrum.values[k]), 1)
    end
    (; energies = spectrum.values, vectors = spectrum.vectors, residuals)
end

"""
    permutation_expectation(sector, psi, i, j)

Return `<psi|P(i,j)|psi>/<psi|psi>`. For a degenerate eigenspace a particular
vector's correlation is basis dependent; compare projected observables instead.
"""
function permutation_expectation(sector::SYTSector, ψ::AbstractVector{Float64}, i, j)
    H = PermutationHamiltonian(sector, [(i, j)])
    image = zeros(sector.dimension)
    apply!(image, H, ψ)
    n2 = dot(ψ, ψ)
    isfinite(n2) && n2 > 0 || throw(ArgumentError("state must have positive finite norm"))
    dot(ψ, image) / n2
end

"""
    dmrg_energy(H, permutation_energy)
    dmrg_correlation(nc, permutation_expectation)

Convert reference values to SUNDMRG's output conventions. SU(2) uses S_i dot S_j;
for nc > 2 the reported energy uses P, while SiSj uses P - 1/nc.
"""
function dmrg_energy(H::PermutationHamiltonian, energy)
    H.sector.nc == 2 ? energy / 2 - sum(b[3] for b in H.bonds; init = 0.0) / 4 : energy
end

function dmrg_correlation(nc, expectation)
    nc = positive_int(nc, "nc")
    nc >= 2 || throw(ArgumentError("nc must be at least two"))
    nc == 2 ? (expectation - 0.5) / 2 : expectation - 1 / nc
end

end
