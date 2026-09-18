# Runtime Options

This page collects runtime controls that are useful once the basic
[`run_DMRG`](@ref) workflow is working: MPI lifecycle management, CPU/GPU
backends, and temporary file-backed storage.

## MPI Lifecycle

By default, `run_DMRG` initializes MPI if it is not already active and finalizes
it afterward only when that run initialized it. MPI initialized by the caller
remains available, including after a failed calculation. The default is convenient
for one calculation per Julia process:

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

did_initialize_mpi = init_DMRG!()
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
    did_initialize_mpi && finalize_DMRG!()
end
```

[`init_DMRG!`](@ref) returns whether it initialized MPI. Call
[`finalize_DMRG!`](@ref) when releasing MPI that your code initialized. When
`manage_mpi = false`, MPI must already be active; the run never finalizes it.
MPI cannot be initialized again in a process after it has been finalized.

Acquired resources are released after both successful and failed calculations.
Storage cleanup and engine finalization are each attempted once; a failure in
either does not prevent finalization of package-owned MPI. If calculation and
cleanup both fail, a `CompositeException` retains both exceptions and their
backtraces, with the calculation failure first. Cleanup details are also logged
because Julia's default aggregate-exception display abbreviates later failures.
When only cleanup fails, its exception is propagated directly.

## Multi-Rank CPU Execution

Every rank must call `run_DMRG` in the same order with matching model, lattice,
schedule, and runtime options. Each rank receives its own rank number; only
rank 0 receives `DMRGOutput`, while other ranks receive `nothing`.

With MPI active, validation and local initialization, storage, result-construction,
and cleanup operations synchronize their failures before the next distributed
phase. The failing rank retains its original exception and backtrace; peers
receive an error naming the phase, failing rank, and original cause. This allows
the tested internal-file failures to clean up on all ranks and leaves caller-owned
MPI available for another calculation. These checkpoints do not provide recovery
from arbitrary exceptions inside numerical kernels or MPI transport failures.

The standalone integration entry point launches fresh one-rank and two-rank
jobs through `MPI.mpiexec()`, using the launcher for the configured MPI library:

```bash
julia --project=. --startup-file=no test/mpi_integration.jl
```

It compares small SU(2) ground and excited singlet energies across process counts,
both Lanczos modes, and memory/JLD2 storage. Ground-state nearest-neighbor
correlations are also compared with independent spin-product-basis references.
Energy, correlation, and unique-ground-state entropy comparisons use
`atol = 1e-10`, `rtol = 0`. Separate failure jobs cover root-only initial/sweep
output failures, cleanup failure, and invalid inputs, including a value invalid
on only one rank. Each MPI job has a 240-second timeout; CI also bounds the
dedicated MPI job independently of the ordinary single-process test suite.

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
For MPI GPU runs, each rank is mapped to a CUDA device using its node-local MPI
rank. On every node, the number of MPI processes must not exceed the number of
CUDA devices visible to each process.

MAGMA initialization and finalization status codes are checked. An initialization
that returns an error status has its acquired MAGMA reference released before the
error is raised; a finalizer that fails is not retried.

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

The `scratch` keyword selects an existing parent directory for temporary storage.
Rank 0 creates a unique subdirectory and removes it at the end of the run,
including failures during initial block output or a sweep. Cleanup only removes
that subdirectory; other files and directories in `scratch` are preserved.
An already-removed temporary directory is harmless. If removal fails, the error
is reported and the remaining engine/MPI cleanup still runs.

File-backed storage is especially useful for table-backed SU(Nc) runs where
environment blocks and tensor operators are reconstructed during sweeps.

## Lanczos Mode

The `alg` keyword controls the ground-state Lanczos vector reconstruction mode:

- `alg = :slow`: reconstruct by replaying the Lanczos recurrence.
- `alg = :fast`: cache Lanczos vectors and reuse them during reconstruction.

The example scripts for larger SU(Nc) calculations use `alg = :fast`. For excited
states, both modes retain the same fully reorthogonalized block Krylov basis.

`lanczos_maxiter` sets the maximum Krylov basis size and defaults to 100. It must
be at least `target + 1`. Excited-state solves start from independent directions
to resolve degenerate levels, and check residuals for the requested state and all
lower Ritz states. Increase `lanczos_maxiter` if the basis limit is too small to
resolve the requested levels. After this check, an already-converged prediction
is retained to avoid rotating between degenerate states on successive sweeps.
A requested level that is not available at an
interior sweep cut raises an error instead of silently returning a lower level.
The ground-state path checks its final residual after refinement, and an
unconverged solve raises an error in either path.

For even-site SU(2) calculations, `target = 0` and `target = 1` refer to the first
and second singlet levels; see [Lanczos Solve](@ref) for the symmetry-sector and
boundary conventions used by the numerical reference tests.

## Correlation Measurements

The `correlation` keyword controls whether a measurement sweep records two-site
correlations:

- `correlation = :none`: no correlation measurements.
- `correlation = :nn`: nearest-neighbor correlations.
- `correlation = :chain`: chain-style correlations with the configured `margin`.

Rank 0 receives the measured values in the `SiSj` field of [`DMRGOutput`](@ref).
