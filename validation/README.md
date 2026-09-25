# SU(N) exact-diagonalization benchmarks

This development utility validates SUNDMRG with full, untruncated SU(N) sectors.
It is not part of the exported package API. Every site carries the fundamental
representation. It provides dense diagonalization for small sectors and an
opt-in block Krylov driver for larger sectors. Both run on the CPU without a
working GPU or starting MPI.

## Design and references

The implementation reuses the mathematical core already present in
[`src/sytx.jl`](../src/sytx.jl): `multiplicity`, `SYTdiagram`, `bf`, and
`Papply2!`. The graph traversal and tableau indexing are already implemented
there and are not reimplemented by the ED utility.

- P. Nataf and F. Mila, [Phys. Rev. B 97, 134420 (2018)](https://doi.org/10.1103/PhysRevB.97.134420):
  standard Young tableaux and Young's orthogonal representation of permutations.
- M. G. Yamada and S. Fujimoto, [Phys. Rev. B 105, L201115 (2022)](https://doi.org/10.1103/PhysRevB.105.L201115):
  tableau indexing through paths in a Wilf-Rao-Shanker graph, and SU(N)-resolved
  computation. Thermal pure quantum states are a future extension, not part of
  this ground/low-energy validation milestone.

For L sites, a sector is a partition of L with at most N rows. Its ED dimension
is the number of standard tableaux, not the dimension of the SU(N) irrep. Keep
all L boxes: a singlet uses the rectangle `(L/N, ..., L/N)`, rather than the
all-zero normalized irrep label used in the DMRG representation layer.

The Hamiltonian is `H_P = sum(J_ij * P_ij)`. Its input is a bond list, including
couplings and duplicate bonds. Nonadjacent exchanges are composed from adjacent
ones. A general exchange need not have only two nonzero entries in a SYT column.

`Papply2!` acts on unit basis vectors to construct adjacent-exchange transitions.
The resulting coefficients are then applied to ordinary Float64 vectors. This
avoids using `SparseVector2` addition, which prunes small amplitudes, during
Hamiltonian applications. The action remains linear at amplitudes below that
pruning threshold. The current transition cache uses O((L-1)*dimension) memory;
graph generation and cache construction are part of setup, not a free operation.

## Independence and validation

The SYT ED shares representation primitives with the coefficient-table builders.
It does not use coefficient tables, DMRG recoupling, truncation, or SUNDMRG's
Lanczos solver, but that alone is not a completely independent check of the SYT
primitives. [`test/reference_sun.jl`](../test/reference_sun.jl) supplies a separate
color-product implementation that swaps site colors directly.

The CPU suite checks:

- symmetric-group relations, including the braid relation, and the complete-graph
  central element against the sum of box contents;
- full product-space spectra reconstructed from all SYT sectors, with each level
  repeated by its SU(N) irrep dimension;
- singlets selected independently by the zero eigenspace of total quadratic
  Casimir (equal color populations alone are insufficient);
- all pair correlations for nondegenerate small singlets, energy residuals,
  duplicate bonds, nonadjacent exchanges, and tiny-amplitude linearity.

`dense_spectrum` returns every level, including repeated eigenvalues. An excited
DMRG target must be compared within the same sector. For degeneracy, use spectral
multiplicities or projected observables; arbitrary eigenvectors, correlations of
individual degenerate vectors, and entropies need not agree. Entanglement requires
a physical bipartition and cannot be obtained by simply reshaping SYT coefficients.

## Output conventions

The ED always reports permutation energies and permutation expectations.
`dmrg_energy` and `dmrg_correlation` explicitly convert these to package outputs:

| Quantity | SU(2) | SU(N), N > 2 |
| --- | --- | --- |
| Energy | `E_P/2 - sum(J)/4` | `E_P` |
| `SiSj` | `(expectation(P_ij) - 1/2)/2` | `expectation(P_ij) - 1/N` |

The N > 2 energy convention includes the constant added by
[`_step_energy`](../src/step_lanczos.jl). The bond correlation has that constant
removed. Use total energies, identical boundaries, and identical multiplicities.
In particular, a width-two square cylinder has two transverse bonds per site pair.
Current DMRG comparisons use unit couplings, as supported by its public model.

## Running a small calculation

Use the repository environment with the MAGMA revision pinned in CI, as for the
ordinary CPU tests. From the repository root:

```bash
julia --project=. --startup-file=no validation/run_sun_ed.jl 3 2 3
julia --project=. --startup-file=no -e 'using Test; include("test/test_sun_ed.jl")'
```

The script computes the singlet of a square cylinder with open x and periodic y
boundaries. It prints the sector, lowest energies, and residuals, without saving
machine-specific metadata. For a custom weighted graph:

```julia
include("validation/sun_ed.jl")
const ED = SUNExactDiagonalization
sector = ED.SYTSector(3, [2, 2, 2])
H = ED.PermutationHamiltonian(sector, [(1, 2, 1.0), (2, 6, 0.5)])
result = ED.dense_spectrum(H)
p12 = ED.permutation_expectation(sector, result.vectors[:, 1], 1, 2)
```

The default sector dimension cap is 100,000, the transition-array payload cap is
512 MiB, and the dense diagonalization cap is 2,048. `sector_spec` checks the
sector and cache limits before constructing the graph. The cache remains a full
O((L-1)*dimension) cache; it rejects oversized requests rather than evicting
transitions. These are allocation guards, not performance promises.

## Block Krylov calculations (ED4)

[`sun_ed_krylov.jl`](sun_ed_krylov.jl) uses
[KrylovKit's BlockLanczos](https://jutho.github.io/KrylovKit.jl/stable/man/algorithms/#KrylovKit.BlockLanczos)
with double reorthogonalization and bounded restarts, independently of SUNDMRG's
Lanczos implementation. KrylovKit belongs to the separate validation environment,
not the package's runtime dependencies. From the repository root:

```bash
julia --project=validation --startup-file=no validation/setup.jl
julia --project=validation --startup-file=no --threads=1 validation/runtests.jl
julia --project=validation --startup-file=no --threads=1 validation/run_sun_ed_krylov.jl 4 4 4 6 8 64 100
```

`setup.jl` resolves the local checkout, CI-pinned MAGMA, and KrylovKit 0.10.4 in a
temporary bootstrap environment, then prepares the ignored validation manifest.
This also works on Julia 1.10, which does not interpret `[sources]`. Run setup
again when switching Julia versions; it does not replace the root manifest or
edit the root project. The separate ED CI job tests minimum and latest Julia.

The CLI arguments are `Nc Lx Ly [levels [blocksize [krylovdim [maxiter]]]]`.
It uses the same square-cylinder bond convention as the dense CLI, including
both transverse bonds at width two. The example has singlet dimension 24,024.
For a custom graph, include both ED files and call:

```julia
include("validation/sun_ed.jl")
include("validation/sun_ed_krylov.jl")
sector = SUNExactDiagonalization.SYTSector(3, [4, 4, 4])
H = SUNExactDiagonalization.PermutationHamiltonian(sector,
    [(i, i + 1) for i in 1:11])
result = SUNEDKrylov.krylov_spectrum(H; levels=3, blocksize=8, krylovdim=64)
```

`levels` counts eigenvalues with multiplicity. The solver calculates up to
`min(d, levels + blocksize)` levels, including guards above the requested cutoff,
and returns the whole cluster intersecting that cutoff. Defaults are seed 1584,
normalized residual tolerance `1e-12`, and cluster width `1e-9 * max(sum(abs(J)),1)`.
Every requested and guard vector must pass independently recomputed residuals,
normalization, and orthogonality checks. `blocksize` must be at least
`min(d, levels+1)`. The implementation rejects an unresolved boundary, a target
cluster saturating the block, or insufficient convergence; increase the block,
Krylov space, or restart limit as appropriate. It never silently falls back to
dense diagonalization or returns unchecked levels.

An initial block cannot resolve arbitrary degeneracy. Random starts, guard
levels, and direct residuals are numerical checks, not a proof of completeness.
For a new benchmark, repeat with another seed and larger block, and compare
cluster projectors or projected observables. The tests compare a nine-fold
eigenspace against the exact projector `(I-P_12)/2`, use independently selected
color-product singlets, and exercise restarted convergence and explicit failures.

The CLI reports setup time (graph and cache), mean time for ten warmed operator
applications, solve time, verification time, and counted operator calls. Package
loading and a tiny compilation warmup are excluded. It also reports cumulative
Julia allocations and retained Hamiltonian bytes. The overflow-safe preflight
estimate includes the transition arrays, multiple simultaneous vector sets, and
projected matrices; the default estimated working-storage cap is 1 GiB. It
allows for a final block beyond `krylovdim`. Neither that estimate nor cumulative
allocations is peak process RSS: graph temporaries, object headers, Julia/BLAS,
allocator behavior, and garbage-collector overhead are excluded. Measure process
memory before increasing limits; a dense H alone would use `8d^2` bytes.

GPU/MPI ED, finite temperature, spatial symmetry, entanglement, and
non-fundamental local representations remain deferred.
