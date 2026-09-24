# Agent Instructions

These instructions apply throughout this repository. Keep durable development
rules here; use the roadmap and changelog for milestone status and run results.

## Project Scope and Sources of Truth

- SUNDMRG.jl implements finite-system DMRG with full SU(N) symmetry, CPU and
  CUDA/MAGMA backends, MPI parallelism, and memory/JLD2 storage.
- Read [README.md](README.md) for supported usage,
  [Project.toml](Project.toml) for dependencies and compatibility,
  [ROADMAP.md](ROADMAP.md) for priorities and release gates, and
  [CHANGELOG.md](CHANGELOG.md) for completed user-visible changes.
- Preserve the public API, `DMRGOutput` fields, coefficient-table format, and
  supported Julia versions in patch work unless the requested scope changes
  those contracts. Follow existing Julia style and English documentation.
- Keep changes focused on the requested milestone or problem. Reproduce
  numerical failures before changing the algorithm; avoid unrelated dependency
  upgrades, formatting changes, or speculative feature work.

## Code Organization

| Area | Main locations |
| --- | --- |
| Public API and result/model types | `src/api.jl`, `src/types.jl`, exports in `src/SUNDMRG.jl` |
| Finite-run orchestration | `src/finite.jl`, `src/finite_config.jl`, `src/finite_phases.jl`, `src/finite_sweep.jl` |
| DMRG steps and eigensolvers | `src/step*.jl`, `src/lanczos.jl`, `src/block.jl`, `src/measurement.jl` |
| Backend, resource ownership, and storage | `src/engine_utils.jl`, `src/runtime.jl`, `src/cleanup.jl`, `src/storage.jl` |
| Representation theory and table generation | `src/representation_theory.jl` and its included files, `utils/`, `jld2/` |
| Regressions and independent references | `test/runtests.jl`, `test/reference_su2.jl`, `test/mpi_integration.jl` |
| User documentation and examples | `docs/src/`, `docs/make.jl`, `examples/` |

Keep backend conversions in engine helpers, file operations in storage helpers,
and lifecycle handling in the runtime/cleanup layers. Check the include order in
`src/SUNDMRG.jl` when adding files or moving definitions.

## Numerical and Mathematical Contracts

- Validate numerical changes against independent small-system references or
  mathematical definitions, not only agreement between two execution paths.
  Reuse the spin-product-basis reference and representation-theory regressions
  where applicable; do not derive expected values from the routine under test.
- Match the symmetry sector, lattice boundaries, bond multiplicities, site
  ordering, and normalization. Even-site SU(2) targets index singlet levels;
  `Sz = 0` alone does not select singlets. Width-two square-lattice cylinders
  retain both periodic bonds between each transverse pair of sites.
- Preserve the SU(2) `S_i · S_j` convention and the general SU(N) bond convention
  `P_ij - 1/N`. Bond dimensions count retained multiplets. See the
  [algorithm](docs/src/algorithm.md) and [usage](docs/src/usage.md) documentation.
- Representation labels are Young row lengths, not Dynkin labels. Consult
  [representation notation](docs/src/representation_notation.md) and
  [Wigner/Racah conventions](docs/src/wigner_racah.md) before changing coefficient
  normalization, permutations, multiplicity indices, or table slots.
- Cover both Lanczos modes when changing the solver. Preserve target ordering,
  residual checks, bounded convergence, and deterministic truncation at ties.
  For degenerate levels, compare invariant subspaces or basis-independent
  quantities across independent runs; vectors and entropies can depend on the
  chosen basis. Preserve convergence checks within each run.
- Use fixed seeds where needed and justify fixture tolerances. The current
  small, untruncated CPU energy references use `atol = 1e-10`; the roadmap
  defines the residual criteria. Do not relax checks just to make a failure pass.

## MPI, Cleanup, and Storage Contracts

- `run_DMRG` returns `(rank, output)` on every rank; only rank 0 receives
  `DMRGOutput`, and other ranks receive `nothing`.
- Finalize MPI only when the operation acquired it. Preserve caller-owned MPI
  on both success and failure, including table builders. MPI cannot be
  initialized again in a process after finalization.
- Register resource ownership when acquired and clean up exactly once, including
  partial initialization. Use the shared cleanup handling to preserve primary
  exceptions and backtraces when cleanup also fails. A storage failure must not
  skip required engine or owned-MPI finalization.
- All ranks must enter matching distributed phases. A `_collective_local`
  callback must contain no MPI communication, and every rank must reach the
  checkpoint. Do not place the checkpoint itself inside a root-only branch.
  The documented recovery covers these local checkpoints, not arbitrary kernel
  exceptions or failed MPI transports.
- Only rank 0 manages package-created scratch directories. Remove only the
  directory owned by the run; preserve unrelated files and tolerate prior
  removal. Keep memory and JLD2 behavior consistent and save GPU data as host
  arrays through the existing helpers.
- Treat the bundled `jld2/` tables as versioned reference data. Keep their schema
  stable and regenerate them only for work that requires it, with provenance
  and coefficient checks. Table generation is a separate workload; use explicit
  arguments and a disposable output directory for small experiments.

## Environment and Verification

Julia 1.10 is the current minimum. [CI](.github/workflows/ci.yml) tests Julia 1.10
and the latest Julia 1.x, with separate ordinary CPU and MPI jobs and a docs job.
Use the active project and match CI's MAGMA revision. For a fresh checkout, run
from the repository root:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/MGYamada/MAGMA.jl.git", rev="5545b1a27ee2516d9766c6a15238f006eceb1629")); Pkg.instantiate()'
```

Keep this pin aligned with CI and installation documentation when deliberately
changing it. Review any resulting `Project.toml` changes before including them.

| Change | Relevant verification from the repository root |
| --- | --- |
| Numerical behavior, API, representation theory, or runtime code | `julia --project=. --startup-file=no -e 'using Pkg; Pkg.test()'` |
| Distributed kernels, runtime ownership, storage, or solver behavior | Also run `julia --project=. --startup-file=no --threads=1 test/mpi_integration.jl` |
| User documentation, docstrings, or docs configuration | `julia --project=docs --startup-file=no docs/make.jl` |
| Only agent instructions or other Markdown outside the docs build | Check referenced paths, command consistency, and `git diff --check` |

For the first docs build, prepare its environment using the same MAGMA pin:

```bash
julia --project=docs --startup-file=no -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/MGYamada/MAGMA.jl.git", rev="5545b1a27ee2516d9766c6a15238f006eceb1629")); Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

- Add focused regression coverage for changed behavior, then run the relevant
  suites. Do not add tests that merely repeat implementation details or rerun
  expensive suites for unrelated prose edits.
- Run the ordinary suite in one process, not under `mpiexec`. The standalone
  MPI driver launches its own workers using `MPI.mpiexec()`. Lifecycle scenarios
  need fresh processes; keep their timeouts and process cleanup bounded.
- Check both CI Julia versions for compatibility-sensitive changes. Report the
  commands, versions, results, and any unavailable configurations accurately;
  historical results and an edited CI file are not evidence of a new run.
- Keep ordinary CPU tests usable without GPU hardware. GPU validation requires
  actual CUDA/MAGMA execution; mocked MAGMA status tests do not establish it.
  For multiple GPUs, verify CUDA-aware MPI and node-local device assignment.
  Record unexecuted hardware configurations as unverified, not passed.
- Follow [.gitignore](.gitignore). Keep generated calculation outputs, temporary
  storage, build output, ignored manifests, and machine-specific preferences out
  of commits unless the task explicitly calls for a reviewed artifact.

## Documentation, Git, and Releases

- Inspect the working tree before editing and preserve unrelated user changes.
  Use `codex/` for agent-created branches unless the user specifies another name.
  Keep commits and PRs focused, with the behavior change, validation evidence,
  and remaining limitations explained for a reviewer.
- When committing, use the user's configured public identity and check that Git
  has not inferred identity metadata from a local machine name. Do not publish
  local connection information in commit metadata.
- Keep public docstrings, usage examples, runtime documentation, and changelog
  entries aligned with changed behavior. Record completed work and actual
  evidence in `ROADMAP.md`; keep outstanding or unavailable checks explicit.
- Preserve the MIT license notices and the attribution in `CITATION.cff` and
  `README.md` when moving or reusing code.
- A request to commit, push, or open a PR does not authorize merging or releasing.
  Keep the changelog's `Unreleased` label until release is explicitly authorized
  and the roadmap's release gates are satisfied. Do not create release tags or
  publish a release as a side effect of implementation work.

## Local Network Configuration

- Do not record the user's local network configuration or connection details in
  repository files, documentation, source comments, test fixtures, generated
  artifacts, saved diagnostic logs, commit messages, issue/PR descriptions,
  or persistent agent notes.
- This includes private IP addresses, subnet/gateway/DNS settings, local
  hostnames, SSIDs, router settings, port forwarding and firewall rules,
  and SSH endpoints, usernames, key paths, and credentials.
- Use connection details only as needed for the current authorized task.
  Use placeholders such as `<GPU_HOST>` and `<SSH_USER>` in saved examples,
  launch commands, and instructions.
- Redact network and connection details before saving diagnostic output or
  publishing validation reports.
- GPU/MPI validation records may include GPU models/counts, OS and software
  versions, numerical results, and commands with placeholders. Omit local
  network setup and remote-access configuration.
