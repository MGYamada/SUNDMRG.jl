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

A lightweight CPU-only unit test suite is available for pure/helper functionality
(e.g., initialization helpers, SU helper routines, sparse vector operations, and
small table helper checks).

Run:
```julia
julia --project -e 'using Pkg; Pkg.test()'
```

## TODO

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
