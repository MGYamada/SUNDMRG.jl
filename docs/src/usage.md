# Usage

This page shows the main user-facing workflow for finite-system DMRG runs.
The examples use small dimensions for clarity; production calculations usually require
larger bond dimensions, multiple MPI ranks, and, for SU(Nc) with ``N_c > 2``,
precomputed coefficient tables.

## Installation

Install the package into a Julia environment with its dependencies available.
MAGMA.jl must be installed before adding SUNDMRG.jl.

```julia
] add https://github.com/MGYamada/MAGMA.jl.git
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

## SU(Nc) Runs With Tables

For ``N_c > 2``, pass a `widthmax` and precomputed coefficient tables.
The repository examples load these from `jld2/table_SU$(Nc)_$(widthmax).jld2`.

```julia
using SUNDMRG
using JLD2

Nc = 3
widthmax = 13
@load joinpath(@__DIR__, "..", "jld2", "table_SU$(Nc)_$widthmax.jld2") tables

rank, dmrg = run_DMRG(
    SU(Nc)HeisenbergModel(),
    HoneycombLattice(6, 6, :ZC),
    (100, 1e-5),
    [(100, 1e-5), (200, 1e-6), (400, 1e-7), (800, 0.0)],
    (1600, 0.0),
    CPUEngine;
    widthmax = widthmax,
    tables = tables,
    fileio = true,
    correlation = :nn,
    margin = 5,
    alg = :fast,
)
```

## MPI Lifecycle

By default, `run_DMRG` initializes and finalizes MPI for one run.
If you want multiple runs in the same Julia process, manage MPI explicitly.

```julia
using SUNDMRG

init_DMRG!()
try
    run_DMRG(SU(2)HeisenbergModel(), SquareLattice(4, 4), 100, [100], 100, CPUEngine; manage_mpi = false)
    run_DMRG(SU(2)HeisenbergModel(), SquareLattice(6, 4), 100, [100], 100, CPUEngine; manage_mpi = false)
finally
    finalize_DMRG!()
end
```

## Common Keywords

- `target = 0`: target state, where `0` is the ground state.
- `widthmax = 0`: representation table width for ``N_c > 2``.
- `tables = nothing`: precomputed table dictionary for ``N_c > 2``.
- `fileio = false`: store intermediate blocks on disk instead of only in memory.
- `scratch = "."`: directory used for temporary file-backed storage.
- `correlation = :none`: set to `:nn` or `:chain` to measure correlations.
- `margin = 0`: boundary margin used by correlation measurement.
- `alg = :slow`: Lanczos mode; examples for larger runs use `:fast`.
- `verbose = true`: print progress on rank 0.
- `manage_mpi = true`: let `run_DMRG` manage MPI for a single call.

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
