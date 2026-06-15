# SUNDMRG.jl

SUNDMRG.jl is a DMRG implementation with full SU(N) symmetry.

This documentation is organized around three common reading paths:
running calculations, understanding the algorithm, and working with the
SU(Nc) representation data used internally by the package.

```@contents
Pages = ["usage.md", "runtime_options.md", "examples.md", "coefficient_tables.md", "dmrg_overview.md", "algorithm.md", "representation_theory.md", "su_n_examples.md", "representation_notation.md", "wigner_racah.md", "api.md"]
Depth = 2
```

## Reading Path

If you want to run a calculation, start with [Usage](usage.md), then continue to
[Runtime Options](runtime_options.md) for MPI, GPU, file-backed storage, and
measurement controls. Use [Examples](examples.md) for the checked-in scripts.

If you want to understand the numerical method, read [DMRG Overview](dmrg_overview.md)
first, then [SUNDMRG Algorithm](algorithm.md) for the package-specific warmup,
growth, sweep, truncation, and measurement flow.

If you are working with coefficient tables, read
[Coefficient Tables](coefficient_tables.md). For the representation labels behind
those tables, read [SU(Nc) Representation Theory](representation_theory.md),
continue to [Examples of SU(Nc) Representations](su_n_examples.md), and keep
[Representation Labels in SUNDMRG.jl](representation_notation.md) nearby for the
row-length convention used in package-facing labels. For the recoupling
coefficients themselves, use [Wigner/Racah Coefficients](wigner_racah.md).

Use [API Reference](api.md) when you need signatures, return fields, and exported
entry points.
