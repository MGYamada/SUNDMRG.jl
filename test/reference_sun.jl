module SUNColorReference

using LinearAlgebra

# An independent color-product basis: no SYT, recoupling, or DMRG routines.
function color_basis(nc, nsites; occupations = nothing)
    states = Vector{Int}[]
    counts = zeros(Int, nc)
    state = zeros(Int, nsites)
    function extend(site)
        if site > nsites
            push!(states, copy(state))
            return
        end
        for color in 1 : nc
            if occupations === nothing || counts[color] < occupations[color]
                state[site] = color
                counts[color] += 1
                extend(site + 1)
                counts[color] -= 1
            end
        end
    end
    extend(1)
    states
end

function hamiltonian(states, bonds)
    indices = Dict(Tuple(state) => i for (i, state) in enumerate(states))
    H = zeros(length(states), length(states))
    for (column, state) in enumerate(states), bond in bonds
        i, j = bond[1], bond[2]
        J = length(bond) == 2 ? 1.0 : bond[3]
        swapped = copy(state)
        swapped[i], swapped[j] = swapped[j], swapped[i]
        H[indices[Tuple(swapped)], column] += J
    end
    H
end

function singlet_reference(nc, nsites, bonds)
    nsites % nc == 0 || throw(ArgumentError("singlets require a multiple of nc sites"))
    states = color_basis(nc, nsites; occupations = fill(nsites ÷ nc, nc))
    H = hamiltonian(states, bonds)
    all_pairs = [(i, j) for i in 1 : nsites for j in i + 1 : nsites]
    # With tr(T^a T^b) = delta_ab/2:
    # C2_total = L*(nc^2-1)/(2nc) + sum_(i<j)(P_ij - 1/nc).
    C2 = hamiltonian(states, all_pairs) +
         (nsites * (nc^2 - 1) / (2nc) - length(all_pairs) / nc) * I
    casimir = eigen(Symmetric(C2))
    singlets = casimir.vectors[:, abs.(casimir.values) .< 1e-10]
    spectrum = eigen(Symmetric(singlets' * H * singlets))
    (; states, H, C2, singlets, energies = spectrum.values, vectors = singlets * spectrum.vectors)
end

# Coordinates are independent of the DMRG growth and bond-building routines.
# Retain both periodic transverse bonds when Ly == 2.
function square_cylinder_bonds(Lx, Ly)
    site(x, y) = (x - 1) * Ly + y
    bonds = Tuple{Int, Int}[]
    for x in 1 : Lx, y in 1 : Ly
        push!(bonds, (site(x, y), site(x, mod1(y + 1, Ly))))
        x < Lx && push!(bonds, (site(x, y), site(x + 1, y)))
    end
    bonds
end

function snake_site(Ly, site)
    x, y = divrem(site - 1, Ly)
    x * Ly + (iseven(x) ? y + 1 : Ly - y)
end

end
