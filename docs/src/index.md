# SUNDMRG.jl

SUNDMRG.jl is a DMRG implementation with full SU(N) symmetry.

This documentation starts with the representation-theoretic background used by the package.

```@contents
Pages = ["dmrg_overview.md", "representation_theory.md", "su_n_examples.md", "representation_notation.md"]
Depth = 2
```

## Reading Path

Start with [DMRG Overview](dmrg_overview.md) for the general algorithmic ideas.
Then read [SU(Nc) Representation Theory](representation_theory.md) for the symmetry notation used in SU(Nc)-adapted calculations.
Continue to [Examples of SU(Nc) Representations](su_n_examples.md) for concrete labels and small tensor-product decompositions.
The short [Representation Labels in SUNDMRG.jl](representation_notation.md) note records the row-length convention used when labels appear in package-facing contexts.
