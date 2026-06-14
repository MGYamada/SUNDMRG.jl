# Runtime Options

This page collects runtime controls that are useful once the basic
[`run_DMRG`](@ref) workflow is working: MPI lifecycle management, CPU/GPU
backends, and temporary file-backed storage.

## MPI Lifecycle

By default, `run_DMRG` initializes MPI before a run and finalizes MPI afterward.
This is convenient for one calculation per Julia process:

```julia
rank, dmrg = run_DMRG(
    SU(2)HeisenbergModel(),
    SquareLattice(4, 4),
    100,
    [100],
    100,
    CPUEngine,
)
```

For multiple calculations in one Julia session, manage MPI outside the individual
calls:

```julia
using SUNDMRG

init_DMRG!()
try
    run_DMRG(
        SU(2)HeisenbergModel(),
        SquareLattice(4, 4),
        100,
        [100],
        100,
        CPUEngine;
        manage_mpi = false,
    )

    run_DMRG(
        SU(2)HeisenbergModel(),
        SquareLattice(6, 4),
        100,
        [100],
        100,
        CPUEngine;
        manage_mpi = false,
    )
finally
    finalize_DMRG!()
end
```

Use [`init_DMRG!`](@ref) and [`finalize_DMRG!`](@ref) as a pair. When
`manage_mpi = false`, MPI must already be initialized.

## CPU And GPU Engines

The final positional argument to `run_DMRG` selects the dense-array backend.

Use [`CPUEngine`](@ref) for CPU execution:

```julia
rank, dmrg = run_DMRG(
    SU(2)HeisenbergModel(),
    SquareLattice(4, 4),
    100,
    [100, 200],
    400,
    CPUEngine,
)
```

Use [`GPUEngine`](@ref) for CUDA-backed execution:

```julia
rank, dmrg = run_DMRG(
    SU(2)HeisenbergModel(),
    SquareLattice(4, 4),
    100,
    [100, 200],
    400,
    GPUEngine,
)
```

GPU runs require CUDA and MAGMA to be configured before starting the calculation.
For MPI GPU runs, the implementation expects the number of MPI ranks to be no
larger than the number of available CUDA devices.

## File-Backed Storage

By default, intermediate blocks, transformation matrices, and tensor data are kept
in memory. For larger runs, use `fileio = true` to store intermediate data in
temporary JLD2 files:

```julia
rank, dmrg = run_DMRG(
    SU(3)HeisenbergModel(),
    HoneycombLattice(6, 6, :ZC),
    100,
    [100, 200, 400],
    800,
    CPUEngine;
    widthmax = widthmax,
    tables = tables,
    fileio = true,
    scratch = "/path/to/scratch",
)
```

The `scratch` keyword selects the parent directory for temporary storage. The
temporary directory is cleaned up at the end of the run on rank 0.

File-backed storage is especially useful for table-backed SU(Nc) runs where
environment blocks and tensor operators are reconstructed during sweeps.

## Lanczos Mode

The `alg` keyword controls the Lanczos vector reconstruction mode:

- `alg = :slow`: reconstruct by replaying the Lanczos recurrence.
- `alg = :fast`: cache Lanczos vectors and reuse them during reconstruction.

The example scripts for larger SU(Nc) calculations use `alg = :fast`.

## Correlation Measurements

The `correlation` keyword controls whether a measurement sweep records two-site
correlations:

- `correlation = :none`: no correlation measurements.
- `correlation = :nn`: nearest-neighbor correlations.
- `correlation = :chain`: chain-style correlations with the configured `margin`.

Rank 0 receives the measured values in the `SiSj` field of [`DMRGOutput`](@ref).
