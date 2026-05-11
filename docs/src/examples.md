# Examples

The repository includes runnable scripts in the `examples/` directory.
Each script accepts `Lx` and `Ly` as command-line arguments, runs DMRG, and saves
the rank-0 result to a JLD2 file.

## CPU SU(2) Square Lattice

`examples/sample_su2_cpu.jl` is the smallest starting point.
It uses on-the-fly SU(2) coefficients and the CPU backend.

```julia
using SUNDMRG
using JLD2

Nc = 2
Lx = 4
Ly = 4
rank, dmrg = run_DMRG(
    SU(Nc)HeisenbergModel(),
    SquareLattice(Lx, Ly),
    100,
    [100, 200, 400, 800],
    1600,
    CPUEngine,
)
```

Run the checked-in script with:

```bash
julia --project examples/sample_su2_cpu.jl 4 4
```

## GPU Variant

The GPU scripts use the same model and lattice setup, but pass `GPUEngine`.
CUDA and MAGMA must be configured before use.

```bash
julia --project examples/sample_su2_gpu.jl 4 4
```

## SU(3), SU(4), And SU(5)

The SU(3), SU(4), and SU(5) examples target a honeycomb lattice with zigzag-cylinder
boundary condition and load precomputed coefficient tables.

```julia
using SUNDMRG
using JLD2

Nc = 3
widthmax = 13
@load joinpath(@__DIR__, "..", "jld2", "table_SU$(Nc)_$widthmax.jld2") tables

Lx = 6
Ly = 6
correlation = Lx < 30 ? :nn : :chain

rank, dmrg = run_DMRG(
    SU(Nc)HeisenbergModel(),
    HoneycombLattice(Lx, Ly, :ZC),
    100,
    [100, 200, 400, 800],
    1600,
    CPUEngine;
    widthmax = widthmax,
    tables = tables,
    fileio = true,
    correlation = correlation,
    margin = 5,
    alg = :fast,
)
```

Representative commands:

```bash
julia --project examples/sample_su3_cpu.jl 6 6
julia --project examples/sample_su4_cpu.jl 6 4
julia --project examples/sample_su5_cpu.jl 10 5
```

Use the corresponding `_gpu.jl` script to run with `GPUEngine`.

## Saving Results

The scripts save `dmrg` only on rank 0:

```julia
if rank == 0
    @save "su$(Nc)_$(Lx)_$Ly.jld2" dmrg
    println("finished!")
end
```

This pattern is important for MPI runs because nonzero ranks return `nothing`
for the result object.

## Table Generation

For ``N_c > 2``, the table files are generated separately.
The table builders are MPI-oriented workloads and are usually run on a cluster.

```julia
using SUNDMRG

make_table3nu(3, 13)
make_table4(3, 13)
make_table(3, 13)
```

`make_table3nu` and `make_table4` produce the partial table files.
`make_table` combines them into the `tables` file consumed by `run_DMRG`.
