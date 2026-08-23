# Changelog

## v1.5.7

### Fixed

- Fixed excited-state Lanczos solves by fully reorthogonalizing their Krylov basis and using Ritz residual convergence, while preserving the lower-memory ground-state path.
- Made Lanczos nested-array helpers accept the abstract vector element types produced by Julia 1.10 DMRG workspaces.
- Validated Lanczos target positions and iteration limits, handled numerical breakdown with a tolerance, and restarted orthogonal Krylov chains when an exact initial eigenvector would otherwise hide later levels.
- Made unavailable Lanczos target levels and unconverged refined wavefunctions raise explicit errors instead of silently returning a lower or unchecked result.
- Made finite-sweep convergence safe when energy or entanglement entropy is zero and bounded cooldown work with `max_cooldown_sweeps`.
- Kept exactly the requested number of density-matrix multiplets when eigenvalues are tied at the truncation cutoff.
- Made unsupported on-the-fly SU(N > 2) 6ν calculations raise a clear `ArgumentError`.
- Corrected coefficient-table documentation so sequential MPI builders share one MPI lifecycle.
- Validated SU(Nc), square-lattice, honeycomb-lattice, and coefficient-table dimensions before runtime initialization.
- Made MPI-owned coefficient-table generation finalize MPI on exceptional exits.
- Assigned CUDA devices by node-local MPI rank and validated the per-node process count.

### Changed

- Bumped the package version to v1.5.7 and set the minimum supported Julia version to 1.10, matching SUNRepresentations 0.3.
- Added CI coverage for both Julia 1.10 and the latest Julia 1.x release.
- Added compatibility bounds for every runtime dependency and pinned the MAGMA.jl revision used by CI and installation documentation.
- Added the `lanczos_maxiter` runtime keyword, with a default of 100.

### Tests

- Added regressions for distinct and restarted Lanczos excited states, unavailable targets, explicit nonconvergence, invalid dimensions and iteration bounds, MPI table lifecycle errors, node-local MPI context, zero-valued convergence data, tied density eigenvalues, cooldown validation, and unsupported SU(3) 6ν evaluation.

## v1.5.6

### Changed

- Bumped package version to v1.5.6.
- Added Wigner/Racah coefficient documentation connecting 3ν, 6ν, and 9ν definitions to implementation functions and DMRG table slots.
- Linked the Wigner/Racah coefficient guide from the documentation navigation, index, and coefficient-table page.

### Tests

- Added definition-based representation-theory tests for Young row-length enumeration, hook-length multiplicities, Weyl dimensions, tensor-product dimension preservation, and SDC orthonormality.
- Added SDC recoupling tests for `_3ν`, `_6ν`, and `_9ν`, including the formal `P_{23}` exchange in the 9ν construction.

## v1.5.5

### Changed

- Bumped package version to v1.5.5.
- Accepted integer density-matrix mixing values in DMRG schedules.
- Added early validation for common `run_DMRG` keyword arguments.

### Tests

- Expanded characterization coverage for API input normalization, MPI lifecycle handling, internal storage, and step helper boundary cases.
- Added small `run_DMRG` regressions for fast Lanczos reconstruction and chain correlation measurement.
- Added itemized unit coverage for `RepresentationTheory` internal helpers.

## v1.5.4

### Changed

- Bumped package version to v1.5.4.
- Added a SUNDMRG-specific algorithm documentation page.
- Reorganized the documentation navigation around getting started, runtime options, coefficient tables, algorithm notes, SU(Nc) symmetry, and API reference.
- Split coefficient-table and runtime-option details out of the usage guide.
- Reduced README duplication by keeping it focused on the shortest runnable example and documentation entry points.

## v1.5.3

### Changed

- Bumped package version to v1.5.3.
- Split SU(N) representation-theory helpers into a `RepresentationTheory` submodule.
- Added SU(2) 3ν, 6ν, and 9ν coefficient regression tests.
- Added SU(3)-SU(5) coefficient regression tests against the bundled JLD2 tables.
- Allowed table-generation utility scripts to accept `Nc` and `widthmax` command-line arguments.
- Made coefficient-table MPI initialization/finalization cooperate with caller-managed MPI sessions.

## v1.5.2

### Changed

- Bumped package version to v1.5.2.
- Enabled Documenter deployment to the `gh-pages` branch from CI.
- Refactored the finite-growth phase into smaller internal helpers.
- Centralized worker placeholder block and environment allocation.

## v1.5.1

### Changed

- Bumped package version to v1.5.1.
- Added GitHub Actions CI for tests and documentation builds.
- Added usage, examples, and API reference documentation pages.
- Added docstrings for the main public model, lattice, engine, MPI lifecycle, and table-generation APIs.

## v1.5.0

### Changed

- Refactored the finite DMRG workflow into smaller internal phase, step, runtime, and workspace helpers.
- Refactored DMRG step and finite-state flow to improve maintainability.
- Replaced long internal DMRG step call sites with structured step request objects.
- Moved MPI lifecycle handling behind `init_DMRG!` and `finalize_DMRG!`, with `run_DMRG` supporting externally managed MPI sessions.
- Added engine-specific allocation helpers and a storage adapter for DMRG block/tensor IO.
- Improved type stability across the DMRG runtime.
- Switched DMRG wavefunction dot products to the local `mydot` helper.

### Fixed

- Fixed precompilation involving CUDA memory type references.
- Fixed SUNIrrep compatibility in `Block` irrep vector typing.
- Fixed SU ambiguity tests and explicit SUNIrrep rank usage.
- Fixed mirror initialization placeholders for right blocks and transfer matrices.
- Fixed missing warmup bond dimension propagation during initialization.
- Fixed growth-phase type parameter capture and generalized `dcinit` invariant tests.

### Tests

- Added and expanded CPU-only unit tests for initialization, step, Lanczos, table, storage, SU helper, sparse vector, and on-the-fly helper functionality.
- Updated README testing notes and TODO entries.

### Documentation

- Added an initial Documenter.jl documentation scaffold.
- Added an SU(Nc) representation theory overview.
- Added concrete SU(2), SU(3), and SU(4) representation examples.
- Added a general DMRG overview page.
- Updated citation and README content.
