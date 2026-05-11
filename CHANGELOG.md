# Changelog

## v1.5.0 - 2026-05-11

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
