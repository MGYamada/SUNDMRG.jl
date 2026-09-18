module SU2DenseReference

using LinearAlgebra

# Physical coordinates, with open x and periodic y boundaries. In particular,
# Ly = 2 has two bonds between the same two sites in each column: keep both.
# The numbering need not follow DMRG's snake path because spectra are invariant
# under a permutation of sites.
function square_cylinder_bonds(Lx, Ly)
    site(x, y) = (x - 1) * Ly + y
    bonds = Tuple{Int, Int}[]
    for x in 1 : Lx, y in 1 : Ly
        push!(bonds, (site(x, y), site(x, mod1(y + 1, Ly))))
        if x < Lx
            push!(bonds, (site(x, y), site(x + 1, y)))
        end
    end
    return bonds
end

# Direct spin-1/2 product-basis construction of H = sum_(i,j) S_i . S_j.
# Work in Sz = 0 to keep the dense reference small. This sector still contains
# every total-spin singlet, together with the Sz = 0 component of higher spins.
function heisenberg_matrix(nsites, bonds)
    basis = [bits for bits in 0 : (1 << nsites) - 1 if count_ones(bits) == nsites ÷ 2]
    indices = Dict(bits => index for (index, bits) in enumerate(basis))
    H = zeros(length(basis), length(basis))
    for (column, bits) in enumerate(basis), (i, j) in bonds
        bit_i = 1 << (i - 1)
        bit_j = 1 << (j - 1)
        if iszero(bits & bit_i) == iszero(bits & bit_j)
            H[column, column] += 1 / 4
        else
            H[column, column] -= 1 / 4
            flipped = xor(bits, bit_i | bit_j)
            H[indices[flipped], column] += 1 / 2
        end
    end
    return H
end

function singlet_reference(nsites, bonds)
    H = heisenberg_matrix(nsites, bonds)
    # S_total^2 = 3N/4 + 2 sum_(i<j) S_i . S_j. Diagonalize this independently
    # of the DMRG irreps/recoupling tables to select total S = 0, not merely Sz = 0.
    all_pairs = [(i, j) for i in 1 : nsites for j in i + 1 : nsites]
    S2 = 2heisenberg_matrix(nsites, all_pairs) + (3nsites / 4) * I
    spin = eigen(Symmetric(S2))
    singlets = spin.vectors[:, abs.(spin.values) .< 1e-10]
    spectrum = eigen(Symmetric(singlets' * H * singlets))
    return (; H, S2, singlets, energies = spectrum.values, vectors = singlets * spectrum.vectors)
end

end
