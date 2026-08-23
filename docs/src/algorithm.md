# SUNDMRG Algorithm

This page explains how SUNDMRG.jl organizes a finite-system DMRG run.
For a general introduction to DMRG, start with [DMRG Overview](dmrg_overview.md).
This page focuses on the package-specific flow, data structures, and symmetry handling.

## Overall Run Flow

The public entry point is [`run_DMRG`](@ref).
After validating the model, lattice, schedule, and keyword arguments, the run builds
an MPI/runtime context and then executes four main phases:

1. Initialize empty left and right blocks.
2. Warm up the system until the transverse width is reached.
3. Grow the finite system to the full lattice size.
4. Sweep through the fixed-size system until convergence, then perform measurements.

Only MPI rank 0 returns a [`DMRGOutput`](@ref). Other ranks participate in the
distributed linear algebra and return `nothing` for the result object.

## Symmetry-Adapted Blocks

SUNDMRG stores each block in SU(Nc) symmetry sectors rather than as one dense Hilbert
space. A block records:

- its length and internal lattice bonds;
- the SU(Nc) irreducible representations present in the block;
- the number of retained multiplets in each representation;
- scalar operators such as the block Hamiltonian;
- tensor operators used to build interactions and measurements.

When a block is enlarged by one site, each current representation is tensored with
the fundamental representation. The resulting candidate representations become the
sector labels of the enlarged block, subject to the representation cutoff used for
precomputed tables.

For SU(2), the required symmetry coefficients can be evaluated on the fly. For
SU(Nc) with `Nc > 2`, production runs usually use precomputed coefficient tables.

## One DMRG Step

A single DMRG step combines an enlarged system block and an enlarged environment block
into a superblock problem. The step performs the following work:

1. Determine the active lattice bonds crossing the current cut.
2. Build the pieces of the effective Hamiltonian in symmetry-sector form.
3. Solve the target superblock state with Lanczos.
4. Build reduced density matrices from the optimized wavefunction.
5. Optionally add density-matrix mixing.
6. Diagonalize the density matrices and choose the retained multiplets.
7. Project the block Hamiltonian and tensor operators into the truncated basis.
8. Return the new block, transformation matrix, energy, truncation error, and
   entanglement data.

The effective Hamiltonian is not materialized as one large dense matrix. Instead,
Lanczos receives a function that applies the Hamiltonian to the structured
wavefunction.

## Wavefunction Layout

The superblock wavefunction is stored as a matrix indexed by environment and system
irreducible representations. Each entry is a vector over outer-multiplicity channels,
and each channel contains a dense matrix of retained multiplet amplitudes.

This layout lets the algorithm apply symmetry selection rules directly. The linear
algebra helpers in the implementation operate on this nested matrix-of-vectors
structure, while MPI reductions combine distributed contributions to inner products,
norms, and Hamiltonian applications.

## Lanczos Solve

The Lanczos solver targets the requested eigenstate of the effective Hamiltonian.
For each Hamiltonian application:

- block Hamiltonians act within their own SU(Nc) sectors;
- inter-block interaction terms use tensor operators on the system and environment;
- SU(Nc) recoupling coefficients connect allowed sector transitions;
- MPI distributes the sector and bond contributions across ranks.

For a ground-state solve, the `alg = :slow` mode reconstructs the target Ritz vector
by replaying the Lanczos recurrence. The `alg = :fast` mode caches Lanczos vectors
on the host and uses them to speed up reconstruction. Excited-state solves retain
and fully reorthogonalize the Krylov basis to prevent duplicate (ghost) Ritz values;
their memory use is therefore higher in either mode. If a Krylov chain ends before
the requested excited level is reached, the solver starts another vector orthogonal
to the retained basis. It never silently substitutes the highest available Ritz
value for a requested level. Interior sweep cuts require the requested level;
boundary cuts may temporarily have a one-dimensional effective space and use the
only available state while the cut moves inward.

The solver checks the final Hamiltonian residual after its refinement step. If the
requested state remains unconverged, it raises an error rather than returning an
unchecked energy. `lanczos_maxiter` controls the Krylov basis limit and must be at
least `target + 1`.

## Density Matrix Truncation

After Lanczos, the optimized wavefunction defines reduced density matrices for the
side being enlarged. These density matrices are block diagonal in SU(Nc) sectors.

SUNDMRG diagonalizes the sector density matrices, ranks the eigenvalues globally,
and keeps the requested number of SU(Nc) multiplets. Because each retained multiplet
represents all states in its irrep, the code also reports the equivalent number of
ordinary states when verbose output is enabled.

The discarded density-matrix weight is recorded as the truncation error. The same
eigenvalues are used to compute the entanglement entropy and, when requested, the
entanglement spectrum.

## Density Matrix Mixing

Each schedule entry can be either `m` or `(m, alpha)`.
The second form enables density-matrix mixing, sometimes called a noise term. Early
sweeps can use a small nonzero `alpha` to reduce the chance of getting trapped in a
poor local minimum. Final sweeps should use zero, or a negligibly small value, so
the reported state is not biased by the mixing term.

## Warmup, Growth, And Sweeps

The warmup phase starts from empty blocks and grows them until the transverse width
`Ly` is reached.

The growth phase then extends the system to the full lattice size `Lx * Ly`. During
growth, the previous optimized wavefunction is transformed into a prediction for the
next step whenever possible.

The sweep phase fixes the system size and moves the active cut left and right through
the lattice. At each cut, the environment block is loaded from saved block data,
enlarged, and used to improve the current system block. Sweeps continue until the
relative changes in energy and entanglement entropy pass the configured tolerances.
Relative changes are well-defined when either value is zero, and
`max_cooldown_sweeps` bounds the number of cooldown sweeps (100 by default). The run
raises an error instead of sweeping indefinitely if that limit is reached.

## Measurements

After convergence, SUNDMRG performs a measurement sweep when correlations were
requested. The main correlation modes are:

- `:none`: no correlation measurement;
- `:nn`: nearest-neighbor bond correlations;
- `:chain`: chain-style correlations using the configured margin.

For SU(Nc), bond expectations are returned in the `P_ij - 1 / N_c` convention.
For SU(2), the Hamiltonian and `SiSj` values use
``\mathbf{S}_i \cdot \mathbf{S}_j``.

## Storage And Backends

Intermediate block data can be kept in memory or written to temporary JLD2 files
with `fileio = true`. File-backed storage is useful for larger sweeps because
environment blocks and tensor operators can be reconstructed from saved data.

The CPU backend stores dense arrays as `Matrix{Float64}`. The GPU backend stores
dense working arrays on CUDA devices and uses MAGMA for symmetric eigenvalue
problems, while file storage converts data back to host arrays.

## Coefficient Tables

The SU(Nc) tensor recoupling data is central to the implementation. SU(2) runs can
compute the required Wigner coefficients as needed. For `Nc > 2`, the table builders
generate coefficient dictionaries ahead of time:

- [`make_table3nu`](@ref) generates the three-symbol table;
- [`make_table4`](@ref) generates the four-index interaction table;
- [`make_table`](@ref) combines those files into the table tuple consumed by DMRG.

The example scripts load bundled tables from the repository's `jld2/` directory.
