# Coefficient Tables

SUNDMRG uses SU(Nc) recoupling coefficients when it builds symmetry-adapted block
operators and superblock interactions. The way those coefficients are supplied
depends on `Nc`.

## SU(2)

SU(2) runs do not need an external coefficient table. The package evaluates the
required Wigner-symbol data on the fly.

```julia
rank, dmrg = run_DMRG(
    SU(2)HeisenbergModel(),
    SquareLattice(4, 4),
    100,
    [100, 200, 400, 800],
    1600,
    CPUEngine,
)
```

The default `widthmax = 0` and `tables = nothing` are appropriate for this case.

## SU(Nc) With `Nc > 2`

For larger symmetry groups, pass both a representation cutoff and a precomputed
table tuple:

```julia
using SUNDMRG
using JLD2

Nc = 3
widthmax = 13
@load joinpath(@__DIR__, "..", "jld2", "table_SU$(Nc)_$widthmax.jld2") tables

rank, dmrg = run_DMRG(
    SU(Nc)HeisenbergModel(),
    HoneycombLattice(6, 6, :ZC),
    100,
    [100, 200, 400, 800],
    1600,
    CPUEngine;
    widthmax = widthmax,
    tables = tables,
)
```

The `widthmax` keyword controls which irreducible representations are retained in
the table-backed representation list. The `tables` keyword supplies the coefficient
dictionaries consumed by the DMRG kernels.

Bundled example tables are available in the repository's `jld2/` directory:

- `table_SU3_13.jld2`
- `table_SU4_9.jld2`
- `table_SU5_3.jld2`

## Generating Tables

For `Nc > 2`, table generation is a separate workload. It is MPI-oriented and is
usually run on a cluster or other multi-process environment.

The generation flow is:

1. [`make_table3nu`](@ref) writes `table3nuhalf_SU$(Nc)_$(widthmax).jld2`.
2. [`make_table4`](@ref) writes `table4half_SU$(Nc)_$(widthmax).jld2`.
3. [`make_table`](@ref) reads those partial files and writes
   `table_SU$(Nc)_$(widthmax).jld2`.

From Julia:

```julia
using SUNDMRG

init_DMRG!()
try
    make_table3nu(3, 13; manage_mpi = false)
    make_table4(3, 13; manage_mpi = false)
finally
    finalize_DMRG!()
end

make_table(3, 13)
```

MPI cannot be initialized again after it has been finalized in the same Julia
process. Therefore, either share one externally managed MPI lifecycle as above or
run each checked-in utility command below in a separate Julia process. Do not call
both MPI table builders sequentially with their default `manage_mpi = true` setting.

From the checked-in utility scripts:

```bash
julia --project=. utils/make_table3nu.jl 3 13
julia --project=. utils/make_table4.jl 3 13
julia --project=. utils/make_table.jl 3 13
```

When table generation is part of a larger MPI-managed Julia session, pass
`manage_mpi = false` to the table builders after initializing MPI externally.
When a builder owns MPI (`manage_mpi = true`), it finalizes MPI even if table
generation or file output raises an exception.

## Table Contents

The combined table loaded by DMRG is a tuple of coefficient dictionaries. Internal
DMRG code uses these tables to update tensor operators, build inter-block
interaction terms, predict wavefunctions across sweeps, and reverse system and
environment orderings.

Most users only need to load the table object and pass it to `run_DMRG`. The
representation-theory details behind the coefficients are described in
[SU(Nc) Representation Theory](representation_theory.md) and
[SUNDMRG Algorithm](algorithm.md). For a direct map from Wigner/Racah symbols to
implementation functions and table slots, see
[Wigner/Racah Coefficients](wigner_racah.md).
