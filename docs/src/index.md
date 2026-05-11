# SUNDMRG.jl

SUNDMRG.jl is a DMRG implementation with full SU(N) symmetry.

This documentation starts with the basic DMRG workflow, then covers package usage,
example scripts, the representation-theoretic background used by the package,
and the public API reference.

```@contents
Pages = ["dmrg_overview.md", "usage.md", "examples.md", "representation_theory.md", "su_n_examples.md", "representation_notation.md", "api.md"]
Depth = 2
```

## Reading Path

Start with [DMRG Overview](dmrg_overview.md) for the general algorithmic ideas.
Use [Usage](usage.md) for the first runnable calls and common options.
The [Examples](examples.md) page explains the scripts in the repository's `examples/` directory.
Then read [SU(Nc) Representation Theory](representation_theory.md) for the symmetry notation used in SU(Nc)-adapted calculations.
Continue to [Examples of SU(Nc) Representations](su_n_examples.md) for concrete labels and small tensor-product decompositions.
The short [Representation Labels in SUNDMRG.jl](representation_notation.md) note records the row-length convention used when labels appear in package-facing contexts.
Use [API Reference](api.md) when you need signatures and return fields.
