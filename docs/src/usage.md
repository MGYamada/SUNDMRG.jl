# Usage

This page shows the main user-facing workflow for finite-system DMRG runs.
The examples use small dimensions for clarity; production calculations usually require
larger bond dimensions, multiple MPI ranks, and backend-specific runtime options.

## Installation

Install the package into a Julia 1.10 or later environment with its dependencies available.
MAGMA.jl must be installed before adding SUNDMRG.jl.

```julia
] add https://github.com/MGYamada/MAGMA.jl.git#5545b1a27ee2516d9766c6a15238f006eceb1629
] add https://github.com/MGYamada/SUNDMRG.jl.git
```

## A First SU(2) Run

SU(2) calculations can evaluate symmetry coefficients on the fly, so no external
table file is needed.

```julia
using SUNDMRG

model = SU(2)HeisenbergModel()
lattice = SquareLattice(4, 4)

rank, dmrg = run_DMRG(
    model,
    lattice,
    100,
    [100, 200, 400, 800],
    1600,
    CPUEngine,
)

if rank == 0
    println(last(dmrg.energies))
end
```

`run_DMRG` returns `(rank, dmrg)`.
The result object is returned only on MPI rank 0; on other ranks, `dmrg` is `nothing`.

## Bond-Dimension Schedule

The warmup, sweep, and cooldown arguments control how many SU(Nc) multiplets are kept.
Each entry can be either an integer `m` or a tuple `(m, alpha)`, where `alpha` is a
density-matrix mixing value.

```julia
m_warmup = (100, 1e-5)
m_sweeps = [(100, 1e-5), (200, 1e-6), (400, 1e-7), (800, 0.0)]
m_cooldown = (1600, 0.0)
```

Density-matrix mixing can help avoid local minima early in a calculation.
Use zero, or a negligibly small value, near the final sweeps.

For more detail on how this schedule is used during warmup, growth, sweeps, and
measurements, see [SUNDMRG Algorithm](algorithm.md).

## SU(Nc) Runs With Tables

SU(2) calculations evaluate their symmetry coefficients on the fly. For
``N_c > 2``, pass a `widthmax` and a precomputed coefficient table with the
`tables` keyword.

See [Coefficient Tables](coefficient_tables.md) for table loading and table
generation. The repository examples load bundled tables from the `jld2/`
directory.

## Common Keywords

- `target = 0`: target state, where `0` is the ground state.
- `lanczos_maxiter = 100`: maximum Krylov basis size; it must be at least `target + 1`.
- `widthmax = 0`: representation table width for ``N_c > 2``.
- `tables = nothing`: precomputed table dictionary for ``N_c > 2``.
- `fileio = false`: store intermediate blocks on disk instead of only in memory.
- `scratch = "."`: directory used for temporary file-backed storage.
- `tol_energy = 1e-5`, `tol_EE = 1e-3`: cooldown convergence tolerances.
- `max_cooldown_sweeps = 100`: stop with an error if cooldown does not converge within this many sweeps.
- `correlation = :none`: set to `:nn` or `:chain` to measure correlations.
- `margin = 0`: boundary margin used by correlation measurement.
- `alg = :slow`: Lanczos mode; examples for larger runs use `:fast`.
- `verbose = true`: print progress on rank 0.
- `manage_mpi = true`: let `run_DMRG` manage MPI for a single call.

See [Runtime Options](runtime_options.md) for MPI lifecycle management, GPU runs,
file-backed storage, Lanczos modes, and correlation measurement options.

## Output

Rank 0 receives a [`DMRGOutput`](@ref) object.
The most commonly inspected fields are:

- `energies`: total energy at recorded steps.
- `errors`: truncation errors.
- `EEs`: entanglement entropy at the active cut for recorded steps.
- `EE`: final entanglement-entropy profile.
- `ES`: entanglement spectrum by SU(Nc) irrep.
- `SiSj`: measured two-site correlations.

For SU(Nc), the two-site bond expectation is returned as ``P_{ij} - 1/N_c``.
For SU(2), the Hamiltonian and `SiSj` use ``\mathbf{S}_i \cdot \mathbf{S}_j``.
