# SUNDMRG.jl

<img src="SUNDMRG.png" width="50%">

SUNDMRG.jl: traditional DMRG code with full SU(N) symmetry implementation.
A part of the code is influenced by Simple DMRG. https://github.com/simple-dmrg/simple-dmrg

## Features

* Fully supporting 2D systems
* Fully supporting MPI parallelization
* Fully supporting CUDA and MAGMA
* CUDA-aware MPI
* File-IO

## Installation

Before `]add`, you must install `MAGMA.jl@0.1.2-`. https://github.com/MGYamada/MAGMA.jl
After that, you can do:
```
]add https://github.com/MGYamada/SUNDMRG.jl.git
```

## Usage

Run a small SU(2) Heisenberg calculation on a 4x4 square lattice with:

```julia
using SUNDMRG

rank, dmrg = run_DMRG(
    SU(2)HeisenbergModel(),
    SquareLattice(4, 4),
    100,
    [100, 200, 400, 800],
    1600,
    CPUEngine,
)
```

`dmrg` is returned only on MPI rank 0. SU(2) coefficients are evaluated on the fly;
SU(N) runs with `N > 2` usually use precomputed coefficient tables.

See the documentation for [usage](docs/src/usage.md), [examples](docs/src/examples.md),
and the [algorithm overview](docs/src/algorithm.md). Runnable scripts are available
in the `examples/` directory.

## Dependency

* Julia 1.10 or later
* CUDA.jl 5 or 6
* MAGMA.jl 0.1.2
* SUNRepresentations.jl 0.3
* MPI.jl: We strongly recommend to use Open MPI.

## Testing

The CPU test suite covers internal helpers, SU(N) coefficients, Lanczos solves,
and small DMRG runs. Numerical regressions use independent dense-matrix references
for clustered/degenerate spectra and SU(2) singlet energies, and check convergence,
cooldown limits, and density-matrix truncation across symmetry sectors.
Runtime regressions inject storage/cleanup failures and launch separate Julia
processes to check MPI ownership for DMRG and coefficient-table builders.
These tests use one MPI rank and do not require GPU hardware.

Run the ordinary suite with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The separate MPI integration suite compares one-rank and two-rank SU(2) runs,
including independent energy/correlation references and coordinated I/O failures:

```bash
julia --project=. --startup-file=no test/mpi_integration.jl
```

## TODO

The [v1.5.8 roadmap](ROADMAP.md) tracks the next patch release, including
priorities and completion criteria. Longer-term feature candidates are:

* Hybrid parallelization
* Supporting the triangular lattice
* Thick-restart Lanczos

## Highly unlikely future features

* Supporting the kagome lattice
* MPS formulation
* Supporting a spin system not with a fundamental representation per site

## Citation

If you write a paper using this code, please cite the following papers as well.

`Masahiko G. Yamada, arXiv:2601.06549 (2026).`

https://arxiv.org/abs/2601.06549

## License

MIT

## Authors

* Masahiko G. Yamada
* James R. Garrison
* Ryan V. Mishmash

Please inquire questions to Masahiko G. Yamada (@MGYamada).
Some functions are written by @maartenvd.
I would also thank Frank Pollmann and Karlo Penc for stimulating discussions.
