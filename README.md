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

If you want to run the simulation of the SU(2) Heisenberg model on the 4x4 square lattice
with a default setting, please use the following.
```julia
rank, dmrg = run_DMRG(SU(2)HeisenbergModel(), SquareLattice(4, 4), 100, [100, 200, 400, 800], 1600, CPUEngine)
```
`dmrg` is returned only when `rank == 0` when you are using the MPI parallelization.

For a more detailed configuration, please look at the examples directory.

Please be careful that the expectation value of the bond operator is returned
in the format of P<sub>ij</sub> - 1 / N<sub>c</sub> for SU(N<sub>c</sub>) for the accuracy.

## SU(2)

SUNDMRG.jl supports an on-the-fly calculation of SU(2) symmetry coefficients (Wigner symbols).
The bond Hamiltonian is **S**<sub>i</sub>・**S**<sub>j</sub> in the SU(2) case, not P<sub>ij</sub> = 2 **S**<sub>i</sub>・**S**<sub>j</sub> + 1 / 2.
SiSj is also evaluated by **S**<sub>i</sub>・**S**<sub>j</sub>, not P<sub>ij</sub> - 1 / 2.

## Density matrix mixing

The density matrix mixing (sometimes called a noise term) is a technique to have a better convergence
in DMRG. It perturbs the density matrix a little, avoiding being stuck in some local minima.
The small perturbation is specified in the second term of the tuple for each sweep as follows.
```julia
rank, dmrg = run_DMRG(SU(3)HeisenbergModel(), SquareLattice(6, 6), (100, 1e-5), [(100, 1e-5), (200, 1e-6), (400, 1e-7), (800, 1e-8)], (1600, 0.0), CPUEngine; widthmax = widthmax, tables = tables)
```
The value has to become zero or a negligibly small value in the last few sweeps.

## Dependency

* Julia 1.6-
* CUDA.jl 5.0.0-
* MAGMA.jl 0.1.2-
* SUNRepresentations.jl 0.1.2-
* MPI.jl: We strongly recommend to use Open MPI.

## TODO

* Hybrid parallelization
* Supporting the triangular lattice
* Thick-restart Lanczos

## Highly unlikely future features

* Supporting the kagome lattice
* MPS formulation
* Supporting a spin system not with a fundamental representation per site

## Open Question

Why do we need
```julia
signfactor = iseven(Nc) ? -1.0 : 1.0
```
in `finite.jl`?

## License

MIT

## Authors

* Masahiko G. Yamada
* James R. Garrison
* Ryan V. Mishmash

Please inquire questions to Masahiko G. Yamada (@MGYamada).
Some functions are written by @maartenvd.
