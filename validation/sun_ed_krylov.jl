module SUNEDKrylov

using LinearAlgebra
using Random
using KrylovKit
using ..SUNExactDiagonalization: PermutationHamiltonian, EDWorkspace, apply!, positive_int

"""
    memory_estimate(dimension, nsites; blocksize, krylovdim)

Estimate numeric-array storage in bytes, using overflow-safe arithmetic. The
working estimate includes cached transitions, simultaneous Krylov/output/residual
vectors, and projected matrices, allowing for one block beyond `krylovdim`.
It is an allocation guard, not a bound on process RSS or cumulative allocations;
graph setup, Julia/BLAS, object headers, and garbage-collector overhead are excluded.
"""
function memory_estimate(dimension, nsites; blocksize, krylovdim)
    d, L, b, k = (positive_int(x, name) for (x, name) in
                  ((dimension, "dimension"), (nsites, "nsites"),
                   (blocksize, "blocksize"), (krylovdim, "krylovdim")))
    b <= k <= d || throw(ArgumentError("require blocksize <= krylovdim <= dimension"))
    capacity = big(k) + b
    transitions = big(d) * (L - 1) * (2sizeof(Float64) + sizeof(Int))
    vectors = sizeof(Float64) * big(d) * (4capacity + 6big(b) + 4)
    projected = sizeof(Float64) * 6capacity^2
    (; transitions, vectors, projected, working = transitions + vectors + projected,
       dense_matrix = sizeof(Float64) * big(d)^2)
end

function positive_tolerance(value, name)
    value isa Real && !(value isa Bool) && isfinite(value) && 0 < value < 1 ||
        throw(ArgumentError("$name must be finite and between zero and one"))
    converted = Float64(value)
    0 < converted < 1 || throw(ArgumentError("$name must fit in Float64"))
    converted
end

function level_clusters(energies, gap)
    clusters = UnitRange{Int}[]
    first = 1
    for k in 2 : length(energies)
        if energies[k] - energies[first] > gap
            push!(clusters, first : k - 1)
            first = k
        end
    end
    push!(clusters, first : length(energies))
    clusters
end

"""
    krylov_spectrum(H; levels=6, blocksize=min(d, max(8, levels+1)),
                    krylovdim=min(d, max(64, 4blocksize)), maxiter=100,
                    tol=1e-12, cluster_tol=1e-9, seed=1584,
                    max_memory_bytes=2^30)

Find the lowest permutation-energy levels with KrylovKit's BlockLanczos. Include
the entire cluster intersecting `levels`, so the output can contain more levels.
Require convergence of `min(d, levels + blocksize)` target and guard levels and
independently check their normalized residuals and orthonormality. Fail on an unresolved cluster
boundary or a cluster saturating the initial block (unless it spans the full
space). Increase the block size for higher degeneracy. A seeded random start and
these checks provide numerical evidence, not a rigorous completeness certificate.

`tol` bounds `norm(H*v-E*v)/max(sum(abs(J)),abs(E),1)` for unit vectors. Cluster
width is at most `cluster_tol * max(sum(abs(J)),1)`. Solve time includes initial
block generation; verification time and actual operator-call counts are separate.
The operator is scaled internally to keep the residual tolerance dimensionless.
"""
function krylov_spectrum(H::PermutationHamiltonian;
                         levels = 6, blocksize = min(H.sector.dimension, max(8, levels + 1)),
                         krylovdim = min(H.sector.dimension, max(64, 4blocksize)),
                         maxiter = 100, tol = 1e-12, cluster_tol = 1e-9, seed = 1584,
                         max_memory_bytes = 2^30)
    d = H.sector.dimension
    levels = positive_int(levels, "levels")
    blocksize = positive_int(blocksize, "blocksize")
    krylovdim = positive_int(krylovdim, "krylovdim")
    maxiter = positive_int(maxiter, "maxiter")
    budget = positive_int(max_memory_bytes, "max_memory_bytes")
    levels <= d || throw(ArgumentError("levels exceeds sector dimension"))
    min(d, big(levels) + 1) <= blocksize <= d ||
        throw(ArgumentError("blocksize must be between min(d, levels+1) and d"))
    requested = Int(min(d, big(levels) + blocksize))
    requested <= krylovdim <= d ||
        throw(ArgumentError("krylovdim must be between min(d, levels+blocksize) and d"))
    # Allow restart room except when the whole sector fits in the Krylov space.
    krylovdim == d || krylovdim >= 2blocksize ||
        throw(ArgumentError("a restarted solve needs krylovdim >= 2blocksize"))
    tol = positive_tolerance(tol, "tol")
    cluster_tol = positive_tolerance(cluster_tol, "cluster_tol")
    cluster_tol >= 10tol || throw(ArgumentError("cluster_tol must be at least 10tol"))
    seed isa Integer && !(seed isa Bool) && 0 <= seed <= typemax(Int) ||
        throw(ArgumentError("seed must be a nonnegative machine integer"))
    memory = memory_estimate(d, H.sector.nsites; blocksize, krylovdim)
    memory.working <= budget || throw(ArgumentError("estimated working storage $(memory.working) exceeds max_memory_bytes=$budget"))

    scale = max(1.0, sum(abs(b[3]) for b in H.bonds; init = 0.0))
    isfinite(scale) || throw(ArgumentError("sum of absolute couplings must be finite"))
    workspace = EDWorkspace(H)
    calls = 0
    function action(v)
        calls += 1
        image = similar(v)
        apply!(image, H, v, workspace)
        image ./= scale
        image
    end
    solve_start = time_ns()
    rng = MersenneTwister(seed)
    initial = KrylovKit.Block([randn(rng, d) for _ in 1 : blocksize])
    # Numerical rank must reject roundoff-sized residual directions when a
    # degenerate block becomes dependent. Machine epsilon alone creates ghosts.
    algorithm = KrylovKit.BlockLanczos(; krylovdim, maxiter, tol = tol / 10,
                                       qr_tol = max(64eps(Float64), tol / 100), verbosity = 0)
    values, vectors, info = KrylovKit.eigsolve(action, initial, requested, :SR, algorithm)
    solve_seconds = (time_ns() - solve_start) / 1e9
    solve_matvecs = calls
    length(values) <= d || error("Krylov basis exceeded the sector dimension")
    info.converged >= requested && length(values) >= requested ||
        error("block Krylov did not converge all $requested requested/guard levels " *
              "(converged=$(info.converged)); increase krylovdim, blocksize, or maxiter")

    verify_start = time_ns()
    energies = values[1 : requested] .* scale
    issorted(energies) && all(isfinite, energies) || error("invalid Krylov energy ordering")
    vectors = vectors[1 : requested]
    residuals = Float64[]
    image = zeros(d)
    for (E, v) in zip(energies, vectors)
        abs(norm(v) - 1) <= 1e-10 || error("Krylov vector is not normalized")
        apply!(image, H, v, workspace)
        @. image -= E * v
        residual = norm(image) / max(scale, abs(E))
        isfinite(residual) && residual <= tol || error("direct residual $residual exceeds tol=$tol")
        push!(residuals, residual)
    end
    orthogonality = maximum(abs(dot(vectors[i], vectors[j]) - (i == j))
                            for i in 1 : requested for j in 1 : requested)
    orthogonality <= 1e-10 || error("Krylov vectors lost orthogonality: $orthogonality")

    clusters = level_clusters(energies, cluster_tol * scale)
    boundary = findfirst(cluster -> levels in cluster, clusters)
    selected = clusters[1 : boundary]
    lastlevel = last(last(selected))
    if requested < d && lastlevel == requested
        error("no resolved gap after the target cluster; increase blocksize")
    end
    if blocksize < d && any(cluster -> length(cluster) >= blocksize, selected)
        error("target cluster saturates blocksize=$blocksize; increase blocksize")
    end
    verification_seconds = (time_ns() - verify_start) / 1e9
    (; energies = energies[1 : lastlevel], vectors = vectors[1 : lastlevel],
       residuals = residuals[1 : lastlevel], clusters = selected,
       guard_energies = energies[lastlevel + 1 : end],
       max_checked_residual = maximum(residuals), orthogonality,
       checked_levels = requested, blocksize, krylovdim, iterations = info.numiter,
       solve_matvecs, verification_matvecs = requested,
       solve_seconds, verification_seconds, memory, seed)
end

end
