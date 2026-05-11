# API Reference

This page lists the public API exported by SUNDMRG.jl.

## Running DMRG

```@docs
run_DMRG
DMRGOutput
```

## Models And Symmetries

```@docs
SU
HeisenbergModel
```

## Lattices

```@docs
SquareLattice
HoneycombLattice
```

## Engines

```@docs
CPUEngine
GPUEngine
```

## MPI Lifecycle

```@docs
init_DMRG!
finalize_DMRG!
```

## Coefficient Tables

```@docs
make_table3nu
make_table4
make_table
```
