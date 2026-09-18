# v1.5.8 Roadmap

Status: in development, unreleased. Package metadata is set to `1.5.8`.
M1-M3 are complete; M4 and M5 remain open.

## Release Goal

Make the numerical and runtime behavior strengthened in v1.5.7 reproducible across
the supported execution paths. Keep the public `run_DMRG` API, output fields,
coefficient-table format, and Julia 1.10 minimum compatible with v1.5.7.
Fixes should be driven by a reproducer and an independent numerical reference
where applicable.

## Starting Point

- v1.5.7 added excited-state Lanczos reorthogonalization and residual checks,
  bounded cooldown sweeps, deterministic truncation counts for tied density
  eigenvalues, and node-local GPU assignment. These are the baseline to preserve.
- The v1.5.7 [CI](.github/workflows/ci.yml) ran the CPU test suite on Julia 1.10
  and the latest Julia 1.x and built the documentation. M3 adds a separate
  multi-rank MPI job; GPU execution remains part of M4.
- [DMRG regressions](test/test_run_dmrg.jl) cover small SU(2) square lattices,
  excited states, both Lanczos modes, file-backed storage, and correlations.
  [Coefficient regressions](test/test_tables_ground_truth.jl) cover bundled
  SU(3)-SU(5) tables; full table-backed DMRG needs separate coverage.
- The v1.5.7 [storage tests](test/test_storage.jl) covered round trips and normal
  cleanup. M2 below records the added initialization-failure and cleanup-failure
  coverage.

## Milestones

P0 items are release blockers. P1 items complete the intended validation scope;
hardware-dependent evidence has a separate gate below. Work proceeds in the
listed order, with GPU environment preparation starting alongside M1.

### M1 — Numerical Reference Regressions (P0)

- [x] Extend [Lanczos tests](test/test_lanczos_helpers.jl) with small symmetric
  matrices having clustered and degenerate spectra. Check both `alg = :slow`
  and `alg = :fast`, normalized vectors, and residuals against dense
  diagonalization. Compare invariant subspaces for degenerate levels rather
  than requiring a particular eigenvector sign or basis.
- [x] Add a small SU(2) DMRG reference derived independently from the lattice
  bonds and Hamiltonian. Match the symmetry sector, boundary conditions, and
  energy normalization when checking `target = 0` and `target = 1`.
- [x] Extend convergence and truncation coverage for near-zero energies and
  entropies, cooldown-limit exhaustion, and ties spanning multiple irreps.
  Preserve explicit failure for unavailable targets and nonconverged solves.

Completion: Julia 1.10 and the latest Julia 1.x pass the new references and all
v1.5.7 regressions. Small, untruncated CPU energy fixtures should agree within
`atol = 1e-10`; Lanczos fixtures should have a normalized residual
`norm(H * ψ - E * ψ) / max(norm(H), abs(E), 1) <= 1e-8`.
Document any fixture-specific tolerance with its numerical justification.

M1 completed on 2026-09-17. The regressions exposed skipped excited-state levels
and unstable choices inside degenerate eigenspaces. The excited solver now uses
a bounded block Krylov basis, checks all lower residuals, and retains an already
converged prediction after verifying its target eigenvalue.

- [Multiplicity and initial-direction tests](test/test_lanczos_multiplicity.jl)
  cover exact/orthogonal guesses and three random seeds in both modes.
- [Independent SU(2) references](test/reference_su2.jl) cover 2x2 and 2x4
  cylinders; [DMRG tests](test/test_run_dmrg.jl) also guard entropy convergence
  for the twofold-degenerate 2x4 excited singlet with a fixed seed and cooldown cap.
- [Numerical boundary tests](test/test_numerical_boundaries.jl) cover near-zero
  convergence, cooldown exhaustion/recovery, and tied truncation across three irreps.

| Local validation | Dependency configuration | Result |
| --- | --- | --- |
| Full `Pkg.test()` on Julia 1.10.12 | CUDA 6.4.0, JLD2 0.6.7 | 1054/1054 passed |
| Full `Pkg.test()` on Julia 1.12.7 | CUDA 6.3.0, JLD2 0.6.5 | 1054/1054 passed |
| Documenter build on Julia 1.12.7 | `julia --project=docs docs/make.jl` | Passed |

Both test environments used SUNRepresentations 0.3.6, MPI 0.20.27, and MAGMA 0.1.2
at revision `5545b1a27ee2516d9766c6a15238f006eceb1629`. These runs used the CPU
backend on macOS arm64 with one MPI rank; multi-rank and GPU execution remain
part of M3 and M4. Independent review found no outstanding M1 issues.

### M2 — Runtime Ownership and Failure Cleanup (P0)

- [x] Exercise failures after temporary storage creation, during initial block
  output, and during a sweep. Ensure each successfully acquired resource has
  a cleanup path even when initialization does not return a complete state.
- [x] Make cleanup robust to an already-removed temporary directory and verify
  that a storage cleanup error cannot prevent required engine/MPI finalization.
  Keep the original calculation failure visible when cleanup also fails.
- [x] Cover package-owned MPI and caller-owned MPI in separate Julia processes,
  including repeated runs with `manage_mpi = false` and exceptional exits.
  Apply the ownership checks to the MPI coefficient-table builders too.

Primary code: [finite.jl](src/finite.jl),
[finite_phases.jl](src/finite_phases.jl), [runtime.jl](src/runtime.jl),
[storage.jl](src/storage.jl), and
[representation_theory.jl](src/representation_theory.jl).

Completion: injected failures leave no package-created scratch directory when
the filesystem permits removal, preserve unrelated scratch contents, finalize
owned resources once, and leave caller-owned MPI usable. Failed cleanup is
reported without concealing the original failure.

M2 completed on 2026-09-17. Storage ownership is recorded before edge-block
construction and output. Storage, engine, and MPI cleanup have separate scopes,
so a failed finalizer is not retried and cannot skip the remaining cleanup.
[Shared cleanup handling](src/cleanup.jl) preserves primary exceptions and
backtraces, combines simultaneous failures, and logs secondary cleanup details.

- [Storage regressions](test/test_storage.jl) cover repeated cleanup and externally
  removed directories while preserving unrelated scratch files and directories.
- [Runtime failure regressions](test/test_runtime_failures.jl) exercise 11 scenarios
  using real CPU calculations and JLD2 files: failure immediately after storage
  acquisition, partial initial output, a sweep write, and single/combined cleanup
  failures. They check exactly-once cleanup and continued use of caller-owned MPI.
- [MPI ownership regressions](test/test_mpi_ownership.jl) launch 13 bounded,
  independent Julia processes for DMRG and both MPI coefficient-table builders.
  They cover successful and exceptional exits, initialization-hook failures,
  engine/MPI cleanup failures, and successful reuse after caller-owned failures.
- [MAGMA ownership regressions](test/test_magma_runtime.jl) check status-code
  failures and reference rollback with a simulated C API. The implementation
  follows [MAGMA's initialization/finalization contract](https://icl.utk.edu/projectsfiles/magma/doxygen/group__magma__init.html).

| Local validation | Dependency configuration | Result |
| --- | --- | --- |
| Full `Pkg.test()` on Julia 1.10.12 | CUDA 6.4.0, JLD2 0.6.7 | 1342/1342 passed |
| Full `Pkg.test()` on Julia 1.12.7 | CUDA 6.3.0, JLD2 0.6.5 | 1342/1342 passed |
| Documenter build on Julia 1.12.7 | `julia --project=docs docs/make.jl` | Passed |

Both full suites include all M1 numerical references and the 13 MPI ownership
process scenarios. They use the same SUNRepresentations, MPI, and pinned MAGMA
versions recorded for M1, on macOS arm64 with one MPI rank. Independent review
and 37 additional cleanup probes found no outstanding M2 issues. MAGMA error
paths were simulated; actual GPU execution remains part of M4. Multi-rank
failure propagation remains part of M3.

### M3 — Multi-Rank CPU Integration (P0)

- [x] Add a dedicated MPI integration entry point and CI job launching a small
  CPU run with two ranks. Keep lifecycle scenarios in separate processes;
  the existing single-process suite assumes rank 0 and finalizes MPI.
- [x] Compare one-rank and two-rank results for a small SU(2) case, including
  energy, measured correlations, and `fileio = true`. Assert that only rank 0
  receives `DMRGOutput` and every other rank receives `nothing`.
- [x] Exercise a controlled root-rank I/O failure and an invalid-input case.
  Ensure all ranks terminate with a useful failure instead of leaving peers
  waiting in collectives. Set a bounded job timeout to detect hangs.

Completion: the MPI job passes on Julia 1.10 and the latest Julia 1.x, agrees
with the CPU reference within documented tolerances, and exits cleanly on both
success and the tested failures. Implement and test M2 ownership behavior before
relying on it in these distributed failure cases.

The [standalone MPI driver](test/mpi_integration.jl) launches one-rank and
two-rank numerical jobs and seven separate two-rank failure jobs through
`MPI.mpiexec()`. Each launch has a 240-second timeout with bounded process
termination; the dedicated CI job has a 20-minute limit and runs on both
Julia 1.10 and the latest Julia 1.x.

- [Numerical workers](test/mpi_integration_worker.jl) compare all eight
  combinations of `:slow`/`:fast`, memory/JLD2 storage, and ground/excited targets
  on the 2x4 SU(2) cylinder. Energies and all 12 ground-state nearest-neighbor
  correlations are checked against independent spin-product-basis references.
  An additional 4x2 ground-state case exercises growth beyond the warmup lattice
  with JLD2 storage. One-rank/two-rank comparisons use `atol = 1e-10`, `rtol = 0`;
  entropy is compared only for the unique ground state.
- [Failure workers](test/mpi_failure_scenarios.jl) trigger actual root-only JLD2
  failures during initial and sweep output, a root-only cleanup failure, and
  invalid input on all ranks or only rank 0. Every caller-owned scenario reuses
  the same MPI communicator and scratch directory for a successful calculation.
  Separate package-owned scenarios verify finalization after both write failures.
- [Collective regression checks](test/mpi_collective_checks.jl) preserve local
  exceptions and backtraces, report the failing rank to peers, handle simultaneous
  failures and `throw(nothing)`, and verify rollback of successful engine
  initialization when another rank fails.

Failure synchronization is limited to coordinated local validation,
initialization, internal-storage, result-construction, and cleanup checkpoints.
It does not claim recovery from arbitrary numerical-kernel exceptions or MPI
transport failures. All ranks must enter `run_DMRG` in the same order with
matching configuration; see [runtime options](docs/src/runtime_options.md).

M3 completed locally on 2026-09-17. All final integration jobs finished without
hitting their timeout, and the ordinary suite and documentation still pass.

| Local validation | Configuration | Result |
| --- | --- | --- |
| Standalone MPI suite on Julia 1.10.12 | MPICH 5.0.1, one/two ranks | 346/346 driver checks passed; all workers passed |
| Standalone MPI suite on Julia 1.12.7 | Open MPI 5.0.10, one/two ranks | 346/346 driver checks passed; all workers passed |
| Full `Pkg.test()` on Julia 1.10.12 | CUDA 6.4.0, JLD2 0.6.7 | 1342/1342 passed |
| Full `Pkg.test()` on Julia 1.12.7 | CUDA 6.3.0, JLD2 0.6.5 | 1342/1342 passed |
| Documenter build on Julia 1.12.7 | `julia --project=docs docs/make.jl` | Passed |

These runs used macOS arm64 and the same SUNRepresentations, MPI.jl, and pinned
MAGMA versions recorded for M1. Across both Julia versions, the maximum absolute
energy difference from independent diagonalization was approximately `3.11e-14`
over the 2x4 and 4x2 fixtures; the maximum ground-state correlation difference
was `2.33e-15`. The corresponding one-rank/two-rank maxima were `2.66e-15` for
energy and `9.44e-16` for correlations, below the required `1e-10` tolerance.

Independent review and eight timeout-termination probes found no outstanding
M3 issues. The GitHub Actions job is configured and its YAML has been checked,
but remote CI has not been run. Passing remote CI remains a release requirement
under M5; actual GPU execution remains part of M4.

### M4 — Table-Backed and GPU Validation (P1)

- [ ] Add one small CPU SU(3) table-backed DMRG regression using a bundled table,
  with an independent small-system energy reference and an explicit
  `widthmax`. Compare memory and JLD2 storage results on a small square lattice.
- [ ] Add a small SU(2) honeycomb `:ZC` reference regression to cover the second
  supported lattice geometry. Keep both CPU fixtures small enough for ordinary
  CI; a combined SU(3)/honeycomb case can follow if its runtime permits.
- [ ] Add an opt-in GPU smoke entry point that compares a small SU(2) run with
  its CPU reference, checks finite observables, and exercises both Lanczos
  modes. Ordinary CPU tests must remain runnable without GPU hardware.
- [ ] On a CUDA/MAGMA host, record single-GPU results and a two-rank/two-GPU
  result, including device assignment and the too-many-ranks error. Record
  Julia, CUDA, MAGMA, MPI, hardware, commands, and numerical tolerances.

Completion: the SU(3) and honeycomb cases pass in CPU CI. Available GPU runs agree
with CPU references within stated tolerances. Multi-node device assignment is
claimed as verified only when a multi-node run has actually been recorded.

### M5 — Documentation and Release Review (P1)

- [ ] Update [runtime options](docs/src/runtime_options.md) and
  [coefficient-table guidance](docs/src/coefficient_tables.md) with the tested
  launch commands, lifecycle rules, and failure behavior from M2-M4.
- [ ] Keep [installation instructions](docs/src/usage.md) and CI aligned on the
  supported Julia versions, compatibility bounds, and pinned MAGMA revision.
- [ ] Update [CHANGELOG.md](CHANGELOG.md) with completed changes only, and
  record validation results and any explicitly deferred hardware checks.

Completion: the documentation build succeeds, commands describe the tested
configurations, and every release claim is backed by a test or a recorded run.

## Release Gate

- [x] M1-M3 are complete, with no unresolved numerical regression, resource
  ownership failure, or hang in the tested MPI scenarios.
- [ ] The M4 CPU regressions and GPU smoke entry point are complete. GPU runs
  have evidence, or the release notes explicitly record which configurations
  remain unverified because hardware was unavailable. Unverified configurations
  do not count as passed checks; a known GPU regression must be resolved.
- [ ] M5 is complete, both Julia CI versions pass, and the docs build passes.
- [ ] Package metadata and changelog agree on v1.5.8. Remove the `Unreleased`
  label only when the release is ready; create the release tag after review.

## Later Releases

The existing feature backlog remains outside this patch release: hybrid
parallelization, triangular-lattice support, and thick-restart Lanczos. Before
choosing a thick-restart design, measure convergence, allocations, and peak memory
for the current `:slow` and `:fast` modes on fixed ground/excited-state fixtures.
Kagome support, an MPS formulation, and non-fundamental on-site representations
remain exploratory items without a target release.
