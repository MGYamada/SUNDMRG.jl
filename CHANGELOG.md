# Changelog

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
