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

## Representation Theory

```@docs
SUNDMRG.RepresentationTheory
SUNDMRG.RepresentationTheory.RepresentationTable
SUNDMRG.RepresentationTheory.irrep
SUNDMRG.RepresentationTheory.irreplist
SUNDMRG.RepresentationTheory.trivialirrep
SUNDMRG.RepresentationTheory.fundamentalirrep
SUNDMRG.RepresentationTheory.adjointirrep
SUNDMRG.RepresentationTheory.outer_multiplicity
SUNDMRG.RepresentationTheory.OM_matrix
SUNDMRG.RepresentationTheory.wigner3ν
SUNDMRG.RepresentationTheory.wigner6ν
SUNDMRG.RepresentationTheory.wigner9ν
```

## Coefficient Tables

```@docs
make_table3nu
make_table4
make_table
```
