# v1.5.8 Roadmap

Status: in development, unreleased. Package metadata is set to `1.5.8`.
M1-M3 and M3.5's ED1-ED4 implementation are complete. The remaining M4 checks
and M5 remain open.

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

### M3.5 — SU(N) Exact-Diagonalization Benchmarks (P1)

Added between MPI integration and table-backed/GPU validation to supply a
SU(N)-resolved reference for M4. See the [ED design and usage](validation/README.md).

Reuse `multiplicity`, `SYTdiagram`, `bf`, and `Papply2!` from
[sytx.jl](src/sytx.jl), including its existing tableau indexing. The design follows
the Young-orthogonal representation in
[Nataf and Mila (2018)](https://doi.org/10.1103/PhysRevB.97.134420) and the
Wilf-Rao-Shanker indexing used by
[Yamada and Fujimoto (2022)](https://doi.org/10.1103/PhysRevB.105.L201115).
The first target is DMRG validation with fundamental representations, not TPQ.

- [x] ED1: add a CPU validation utility using full L-box Young diagrams and
  weighted bond lists. Extract adjacent-exchange transitions from the existing
  SYT routines; apply them to dense vectors without sparse amplitude pruning.
  Preserve duplicate bonds and compose arbitrary exchanges explicitly.
- [x] ED2: validate against direct color-product swaps, SU(N) irrep degeneracy
  counts, Casimir-selected singlets, and symmetric-group identities. Provide
  bounded dense diagonalization, direct residuals, and pair expectations.
- [x] ED3: compare the SU(3) six-site square-cylinder singlet spectrum and ground
  correlations with table-backed DMRG, with both Lanczos modes and memory/JLD2
  storage. Record the permutation-energy versus centered-correlation conventions.
- [x] ED4: add an opt-in independent block Krylov driver for larger sectors,
  including degenerate clusters, direct residuals, and setup/application/solve
  timings and memory estimates. Do not reuse the DMRG Lanczos solver.

The shared SYT primitives are also used by table generation, so SYT ED alone
does not independently validate them. Keep a separate small color-product
reference. Select the same irrep for excited targets; use multiplicities or
projected observables for degeneracy. EE/ES require a separate physical
bipartition implementation and are deferred, together with GPU/MPI ED, TPQ,
spatial-symmetry reduction, and non-fundamental on-site representations.

Completion for v1.5.8: ED1-ED3 pass ordinary CPU CI with small untruncated energy
and correlation comparisons at `atol = 1e-10`; ED-only small residuals should be
below `1e-12`. ED4 runs in its own opt-in environment and separate CI job. The
validation utility adds no exported package API or package runtime dependency.

ED1-ED3 implemented on 2026-09-24. The [ED regression suite](test/test_sun_ed.jl)
checks full product-space spectra for SU(2), SU(3), and SU(4), and independently
projects small singlets with the total quadratic Casimir.
The [SU(3) DMRG regression](test/test_sun_ed_dmrg.jl) uses a 2x3 square cylinder,
the bundled SU(3) table, `widthmax = 6`, and 243 retained multiplets to avoid
truncation. It checks targets 0 and 1 for both Lanczos and storage modes, plus
all nine nearest-neighbor correlations for the unique ground state. No
coefficient tables or production solver routines were changed.

| Local validation | Configuration | Result |
| --- | --- | --- |
| Focused ED and SU(3) DMRG tests on Julia 1.12.7 | CPU, one MPI rank for DMRG | 203/203 ED checks and 77/77 DMRG checks passed |
| Full `Pkg.test("SUNDMRG")` on Julia 1.10.12 | Isolated environment, CUDA 6.4.0, JLD2 0.6.7 | 1622/1622 passed |
| Full `Pkg.test()` on Julia 1.13.0 | CUDA 6.3.0, JLD2 0.6.5 | 1622/1622 passed |
| Documenter build on Julia 1.13.0 | CI-pinned MAGMA revision | Passed; deployment skipped |

Both full suites used SUNRepresentations 0.3.6, MPI.jl 0.20.27, and MAGMA 0.1.2
at revision `5545b1a27ee2516d9766c6a15238f006eceb1629`, on macOS arm64.
The existing local manifest references newer Julia standard libraries, so the
minimum-version test used a temporary environment without replacing that manifest:

```bash
julia +1.10 --project=. --startup-file=no --threads=1 -e 'using Pkg; repo = pwd(); Pkg.activate(mktempdir()); Pkg.add(PackageSpec(url="https://github.com/MGYamada/MAGMA.jl.git", rev="5545b1a27ee2516d9766c6a15238f006eceb1629")); Pkg.develop(PackageSpec(path=repo)); Pkg.test("SUNDMRG")'
```

The command-line example `validation/run_sun_ed.jl 3 2 3` also passed on Julia
1.12.7: the five-dimensional singlet sector gives permutation ground energy
`-5.605551275463979` with a normalized residual of approximately `1.20e-15`.
These are CPU results. They do not establish GPU execution or remote CI results.

ED4 implemented on 2026-09-24 in
[`validation/sun_ed_krylov.jl`](validation/sun_ed_krylov.jl), using KrylovKit
0.10.4 BlockLanczos with double reorthogonalization and bounded restarts.
Requested and guard levels undergo direct residual and orthogonality checks;
the target's entire cluster is returned. An unresolved cluster boundary,
block-saturating degeneracy, or failed convergence raises an error. Random block
starts and these checks are numerical evidence, not a completeness proof.
Transition payload and estimated solver memory are checked before their major
allocations; the full cache is retained only when within its budget.

The [separate ED4 tests](validation/runtests.jl) cover independently projected
color-product singlets, a nine-fold eigenspace against `(I-P_12)/2`, two random
seeds, actual restarts, zero operators, insufficient blocks, and allocation guards.
A rank-deficient block regression reproduced spurious directions with an
epsilon-only QR cutoff; the cutoff now rejects roundoff-sized directions while
direct residual acceptance remains `1e-12`.

The warmed [benchmark driver](validation/run_sun_ed_krylov.jl) was run on macOS
arm64 with Julia 1.13.0, KrylovKit 0.10.4, one Julia thread, and one BLAS thread.
For an SU(4) 4x4 square cylinder (28 bonds, singlet shape `[4,4,4,4]`, dimension
24,024), `levels=6`, `blocksize=8`, `krylovdim=64`, `maxiter=100`, seed 1584:

| Measurement | Result |
| --- | --- |
| Permutation ground energy | `-16.96325620698734` |
| Maximum direct residual, including all 14 checked levels | `8.93e-14` |
| Setup / mean Hamiltonian application | 9.176 s / 2.011 ms |
| Solve / direct verification | 4.982 s / 0.031 s |
| Krylov cycles / solve applications | 45 / 1,273 |
| Estimated numeric working storage / retained Hamiltonian | 74,242,752 / 8,654,640 bytes |
| Dense Hamiltonian alone (not allocated) | 4,617,220,608 bytes |
| Cumulative setup / solve-and-verification allocations | 1,040,656,384 / 1,104,355,064 bytes |

Times exclude package loading and a tiny compilation warmup. Cumulative
allocations and working-array estimates are not peak RSS; graph temporaries,
Julia/BLAS, object headers, and garbage collection need additional memory.
An independent-start check on Julia 1.10.12 used seed 1585, `blocksize=12`, and
`krylovdim=96`. All six target energies agreed within `1.32e-13`; the maximum
direct residual over 18 checked levels was `7.03e-14` (38 Krylov cycles).
The root DMRG dependencies and production solver remain unchanged.
The separate ED CI job is configured for Julia 1.10 and latest Julia; remote CI
and GPU/MPI ED execution are not established by these local CPU results.

| ED4 local validation | Result |
| --- | --- |
| `validation/runtests.jl`, Julia 1.10.12 | 54/54 passed |
| `validation/runtests.jl`, Julia 1.13.0 | 54/54 passed |
| Full `Pkg.test("SUNDMRG")`, Julia 1.10.12 validation environment | 1626/1626 passed |
| Full `Pkg.test()`, Julia 1.13.0 repository environment, sequential retry | 1626/1626 passed |
| `test/mpi_integration.jl`, Julia 1.13.0, one and two ranks | 346/346 passed |
| `docs/make.jl`, Julia 1.13.0 | Passed; deployment skipped |

The four additional ordinary-suite checks exercise transition-cache preflight
limits and the dimension/payload estimate without building a large sector.
The documented CLI also completed on Julia 1.13.0 for SU(3), 2x3, including all
five singlet levels with maximum normalized residual `1.94e-15`. During local
environment switching, Julia 1.13 emitted dependency precompile-cache diagnostics;
a fresh CLI run after setup completed without those diagnostics.
The initial Julia 1.13 ordinary-suite run overlapped other environment
precompilation: `caller-owned` and `dmrg-owned-success` each exceeded the existing
180-second child-process limit (1620 passed, six timeout/exit/marker checks
failed). After the other Julia jobs finished, the same full-suite command passed
1626/1626 in 3m42s. No timeout, numerical tolerance, or production code was changed
to obtain that pass. The standalone MPI suite passed 346/346 independently.

### M4 — Table-Backed and GPU Validation (P1)

- [x] Add one small CPU SU(3) table-backed DMRG regression using a bundled table,
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

### M5 — Mathematical Documentation and Release Review (P1)

Prepare the representation-theory implementation for reuse and maintenance by
connecting mathematical objects, conventions, code, and validation evidence.
Start from the [mathematical assets guide](docs/src/mathematical_assets.md), and
extend the existing representation and Wigner/Racah pages where they already
describe the relevant convention. Preserve the runtime/release audit below.

#### M5.1 — Inventory and Writing Plan

- [x] Inventory representation labels and multiplicities, SYT graphs and
  permutation actions, sparse coefficient arithmetic, subduction/gauge fixing,
  recoupling, coefficient tables, and symmetry-resolved ED.
- [x] Add a docs entry point with a reading order, implementation/test map, and
  a common outline for future mathematical notes. Distinguish public interfaces
  from internal routines and development-only utilities.
- [x] Separate definition-based checks, independent small-system references,
  and shared-implementation consistency checks. List documentation gaps without
  presenting finite numerical tests as general mathematical proofs.

M5.1 prepared on 2026-09-25. This is an inventory and writing scaffold; the
detailed notes in M5.2 remain outstanding. Validation:
`julia --project=docs --startup-file=no docs/make.jl` passed on Julia 1.13.0
(deployment skipped). Local documentation links and all 25 source/test file
references in the new guide were checked; `git diff --check` passed. No numerical
code changed, so the numerical suites were not rerun for this preparation.

#### M5.2 — First Mathematical Notes

- [ ] Extend [representation labels](docs/src/representation_notation.md) to
  connect normalized SU(N) labels with full L-box SYT shapes. Distinguish irrep
  dimension, tableau count, tensor-product outer multiplicity, and retained
  DMRG multiplets, using a small example already covered by tests.
- [ ] Write the first SYT/indexing note: explain `multiplicity`, `SYTdiagram`,
  `bf`, and `subdiagram`; define `V`, `E`, `D`, `B`, `F`, the graph/path ordering,
  index origins, and integer-type assumptions. Trace a small tableau through
  its index and an adjacent transposition, with a reproducible CPU example.
- [ ] Document the permutation-action and sparse-arithmetic contracts: action
  order, mutation/aliasing, normalization, amplitude pruning, and the ED path
  that avoids pruning. State each identity's mathematical basis and link the
  associated implementation checks.
- [ ] Add a concise subduction contract note covering input embeddings,
  outer-multiplicity axes, normalization, permutation options, and gauge choice.
  Connect it to the existing [recoupling/table map](docs/src/wigner_racah.md)
  without duplicating that map. Identify any unreviewed phase or ordering claim
  explicitly instead of filling it in from numerical agreement alone.

For v1.5.8 these notes establish conventions and worked examples for the
existing implementation. Exhaustive derivations and a broad example catalog
are follow-up documentation work. New public mathematical APIs, algorithm
changes, and table regeneration are outside this documentation milestone.

#### M5.3 — Runtime Documentation and Release Audit

- [ ] Update [runtime options](docs/src/runtime_options.md) and
  [coefficient-table guidance](docs/src/coefficient_tables.md) with the tested
  launch commands, lifecycle rules, and failure behavior from M2-M4.
- [ ] Keep [installation instructions](docs/src/usage.md) and CI aligned on the
  supported Julia versions, compatibility bounds, and pinned MAGMA revision.
- [ ] Update [CHANGELOG.md](CHANGELOG.md) with completed changes only, and
  record validation results and any explicitly deferred hardware checks.

Completion: the inventory and first mathematical notes are navigable from the
docs, their worked examples have checked outputs and stated conventions, and
the documentation build succeeds. Commands describe tested configurations;
every release claim is backed by a test or a recorded run. The existing M4
hardware requirements and explicit authorization for release remain in force.

## Release Gate

- [x] M1-M3 are complete, with no unresolved numerical regression, resource
  ownership failure, or hang in the tested MPI scenarios.
- [x] M3.5 ED1-ED3 pass, including an independent color-product reference and
  the small SU(3) DMRG comparison. ED4 is also implemented in a separate environment.
- [ ] The M4 CPU regressions and GPU smoke entry point are complete. GPU runs
  have evidence, or the release notes explicitly record which configurations
  remain unverified because hardware was unavailable. Unverified configurations
  do not count as passed checks; a known GPU regression must be resolved.
- [ ] M5's mathematical inventory/first notes and runtime/release audit are
  complete, both Julia CI versions pass, and the docs build passes.
- [ ] Package metadata and changelog agree on v1.5.8. Remove the `Unreleased`
  label only when the release is ready; create the release tag after review.

## Later Releases

The existing feature backlog remains outside this patch release: hybrid
parallelization, triangular-lattice support, and thick-restart Lanczos. Before
choosing a thick-restart design, measure convergence, allocations, and peak memory
for the current `:slow` and `:fast` modes on fixed ground/excited-state fixtures.
Kagome support, an MPS formulation, and non-fundamental on-site representations
remain exploratory items without a target release.
