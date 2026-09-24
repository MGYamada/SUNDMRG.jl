include("sun_ed.jl")

function main(args)
    length(args) == 3 || throw(ArgumentError("usage: run_sun_ed.jl Nc Lx Ly"))
    nc, Lx, Ly = parse.(Int, args)
    nc >= 2 && Lx >= 1 && Ly >= 2 || throw(ArgumentError("require Nc >= 2, Lx >= 1, Ly >= 2"))
    nsites = Base.checked_mul(Lx, Ly)
    nsites % nc == 0 || throw(ArgumentError("site count must be divisible by Nc for a singlet"))
    sector = SUNExactDiagonalization.SYTSector(nc, fill(nsites ÷ nc, nc))
    site(x, y) = (x - 1) * Ly + y
    bonds = Tuple{Int, Int}[]
    for x in 1 : Lx, y in 1 : Ly
        push!(bonds, (site(x, y), site(x, mod1(y + 1, Ly))))
        x < Lx && push!(bonds, (site(x, y), site(x + 1, y)))
    end
    H = SUNExactDiagonalization.PermutationHamiltonian(sector, bonds)
    result = SUNExactDiagonalization.dense_spectrum(H)
    println("SU($nc), square cylinder $Lx x $Ly, singlet shape $(sector.shape)")
    println("dimension=$(sector.dimension), bonds=$(length(bonds)), convention=sum(P_ij)")
    for k in 1 : min(6, sector.dimension)
        println("level=$(k - 1), energy=$(result.energies[k]), residual=$(result.residuals[k])")
    end
end

main(ARGS)
