include("sun_ed.jl")
include("sun_ed_krylov.jl")

using LinearAlgebra
using Random
using Printf

const ED = SUNExactDiagonalization
const EK = SUNEDKrylov

function main(args)
    3 <= length(args) <= 7 || throw(ArgumentError(
        "usage: run_sun_ed_krylov.jl Nc Lx Ly [levels [blocksize [krylovdim [maxiter]]]]"))
    nc, Lx, Ly = parse.(Int, args[1 : 3])
    nc >= 2 && Lx >= 1 && Ly >= 2 || throw(ArgumentError("require Nc >= 2, Lx >= 1, Ly >= 2"))
    nsites = Base.checked_mul(Lx, Ly)
    nsites % nc == 0 || throw(ArgumentError("site count must be divisible by Nc for a singlet"))
    shape = fill(nsites ÷ nc, nc)
    spec = ED.sector_spec(nc, shape)
    d = spec.dimension
    levels = length(args) >= 4 ? parse(Int, args[4]) : min(6, d)
    blocksize = length(args) >= 5 ? parse(Int, args[5]) : min(d, max(8, levels + 1))
    krylovdim = length(args) >= 6 ? parse(Int, args[6]) : min(d, max(64, 4blocksize))
    maxiter = length(args) >= 7 ? parse(Int, args[7]) : 100
    memory = EK.memory_estimate(d, nsites; blocksize, krylovdim)
    memory.working <= 2^30 || throw(ArgumentError("estimated working storage exceeds 1 GiB"))
    println("SU($nc), square cylinder $Lx x $Ly, singlet shape $shape")
    println("dimension=$d, levels=$levels, blocksize=$blocksize, krylovdim=$krylovdim, maxiter=$maxiter, seed=1584")
    println("Julia=$VERSION, KrylovKit=$(pkgversion(EK.KrylovKit)), Julia threads=$(Threads.nthreads()), BLAS threads=$(BLAS.get_num_threads())")
    println("estimated numeric working bytes=$(memory.working), transition bytes=$(memory.transitions), dense H bytes=$(memory.dense_matrix)")

    # Compile the same concrete operator/solver types on a tiny sector. The
    # following measurements exclude package loading and this one-time warmup.
    warmup = ED.PermutationHamiltonian(ED.SYTSector(2, [2, 2]), [(1, 2), (2, 3)])
    EK.krylov_spectrum(warmup; levels = 1)
    setup = @timed begin
        sector = ED.SYTSector(nc, shape)
        site(x, y) = (x - 1) * Ly + y
        bonds = Tuple{Int, Int}[]
        for x in 1 : Lx, y in 1 : Ly
            push!(bonds, (site(x, y), site(x, mod1(y + 1, Ly))))
            x < Lx && push!(bonds, (site(x, y), site(x + 1, y)))
        end
        ED.PermutationHamiltonian(sector, bonds)
    end
    H = setup.value
    println("bonds=$(length(H.bonds)), convention=sum(J_ij * P_ij)")
    @printf("setup seconds=%.6f, cumulative allocated bytes=%d, retained H bytes=%d\n",
            setup.time, setup.bytes, Base.summarysize(H))
    v = normalize!(randn(MersenneTwister(1584), d))
    image = similar(v)
    workspace = ED.EDWorkspace(H)
    ED.apply!(image, H, v, workspace)
    application = @timed for _ in 1 : 10
        ED.apply!(image, H, v, workspace)
    end
    @printf("mean application seconds=%.9f, allocated bytes per application=%.1f\n",
            application.time / 10, application.bytes / 10)

    solve = @timed EK.krylov_spectrum(H; levels, blocksize, krylovdim, maxiter)
    result = solve.value
    @printf("solve seconds=%.6f, verification seconds=%.6f, cumulative allocated bytes=%d\n",
            result.solve_seconds, result.verification_seconds, solve.bytes)
    println("iterations=$(result.iterations), solve matvecs=$(result.solve_matvecs), verification matvecs=$(result.verification_matvecs)")
    println("checked levels=$(result.checked_levels), max direct residual=$(result.max_checked_residual), orthogonality error=$(result.orthogonality)")
    for (cluster, indices) in enumerate(result.clusters), k in indices
        println("level=$(k - 1), cluster=$cluster, energy=$(result.energies[k]), residual=$(result.residuals[k])")
    end
    println("guard energies=$(result.guard_energies)")
    println("Memory figures are estimates/allocations, not peak RSS; random-block checks are not a completeness proof.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
