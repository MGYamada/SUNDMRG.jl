# SU(Nc) Representation Theory

This page summarizes the representation-theoretic notation used by SUNDMRG.jl.
It deliberately stays at the level of SU(Nc) representation theory and does not discuss the DMRG algorithm.

The main objects are irreducible representations, Young diagrams, Dynkin labels, tensor-product decompositions, conjugate representations, singlets, and the general structure of symmetry-adapted matrix elements.

## SU(Nc) And Irreducible Representations

SU(Nc) is the Lie group of ``N_c \times N_c`` unitary matrices with determinant one.
Its Lie algebra ``\mathfrak{su}(N_c)`` has ``N_c^2 - 1`` generators.

A vector space used in a quantum many-body problem often carries a representation of SU(Nc).
This means that each group element ``g \in SU(N_c)`` acts as a linear map ``D(g): V \to V``.
An irreducible representation is a representation that cannot be decomposed into smaller nontrivial invariant subspaces.

Finite-dimensional irreducible representations of SU(Nc) can be labeled by highest weights, Dynkin labels, or Young diagrams.
These are equivalent descriptions of the same objects, but each is convenient for a different calculation.

## Fundamental Representation

The fundamental representation of SU(Nc) has dimension ``N_c``.
In Young-diagram notation it is represented by a single box.
In this documentation we describe Young diagrams by their row lengths, so the fundamental representation is written as ``(1)``.

The conjugate of the fundamental representation is the antifundamental representation.
For SU(2), the fundamental and antifundamental representations are equivalent.
For SU(Nc) with ``N_c > 2``, they are generally distinct.

## Young Diagrams

A Young diagram is a left-justified collection of boxes arranged in rows.
Let the row lengths be

```math
\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_{N_c - 1} \ge 0.
```

Then ``(\lambda_1, \lambda_2, \ldots, \lambda_{N_c-1})`` labels an SU(Nc) irreducible representation.
For SU(Nc), a column of height ``N_c`` is equivalent to the trivial representation and can be removed, so one only needs diagrams with at most ``N_c - 1`` rows.

Young diagrams encode two elementary symmetrization rules:

- Indices in the same row are symmetrized.
- Indices in the same column are antisymmetrized.

For example, the tensor product of two fundamental representations decomposes into a symmetric two-box representation and an antisymmetric two-box representation:

```math
\mathbf{N_c} \otimes \mathbf{N_c}
  = \mathrm{Sym}^2(\mathbf{N_c}) \oplus \wedge^2(\mathbf{N_c}).
```

In row-length notation, these are ``(2)`` and ``(1, 1)``.

## Dynkin Labels

The Dynkin labels ``[a_1, a_2, \ldots, a_{N_c-1}]`` are related to Young-diagram row lengths by

```math
a_i = \lambda_i - \lambda_{i+1},
```

with ``\lambda_{N_c} = 0``.
Conversely,

```math
\lambda_i = \sum_{j=i}^{N_c-1} a_j.
```

For SU(3), Dynkin labels are often written as ``[p, q]``.
The fundamental representation is ``[1, 0]``, the antifundamental representation is ``[0, 1]``, and the adjoint representation is ``[1, 1]``.

## Dimension Formula

The dimension of the SU(Nc) irreducible representation associated with a Young diagram ``\lambda`` is given by the hook-length formula.
For a box ``(i, j)`` in row ``i`` and column ``j``, let ``h(i,j)`` be its hook length.
Then

```math
\dim(\lambda)
  = \prod_{(i,j) \in \lambda}
    \frac{N_c + j - i}{h(i,j)}.
```

This gives ``\dim(1) = N_c`` for the fundamental representation and ``N_c(N_c-1)/2`` for the two-index antisymmetric representation ``(1,1)``.

## Tensor-Product Decomposition

When two SU(Nc) representation spaces are combined, their tensor product decomposes into a direct sum of irreducible representations:

```math
\alpha \otimes \beta
  = \bigoplus_{\gamma} N_{\alpha\beta}^{\gamma}\, \gamma.
```

The integer ``N_{\alpha\beta}^{\gamma}`` is a Littlewood-Richardson coefficient.
It counts how many times the irreducible representation ``\gamma`` appears in the tensor product ``\alpha \otimes \beta``.

Tensoring with the fundamental representation can be understood as adding one box to a Young diagram.
The resulting diagram must still have nonincreasing row lengths, and any column of height ``N_c`` can be removed according to the SU(Nc) equivalence.

## Conjugate Representations

The conjugate representation ``\lambda^\ast`` reverses the SU(Nc) weight structure.
In Dynkin-label notation, conjugation reverses the order of the labels:

```math
[a_1, a_2, \ldots, a_{N_c-1}]^\ast
  = [a_{N_c-1}, \ldots, a_2, a_1].
```

For SU(3), this gives ``[1,0]^\ast = [0,1]``.
For SU(2), there is only one Dynkin label, so every irreducible representation is self-conjugate.

## Singlet Condition

The trivial representation, or singlet, is denoted by ``\mathbf{1}``.
Singlets are important because they are invariant under the full SU(Nc) action.

A basic singlet relation is

```math
\alpha \otimes \alpha^\ast \supset \mathbf{1}.
```

More generally, deciding whether a product of representations can form an invariant state is a question about whether the trivial representation appears in the corresponding tensor-product decomposition.

## Irreducible Tensor Operators

Operators can also be classified by how they transform under SU(Nc).
An operator transforming as an irreducible representation ``\gamma`` is an irreducible tensor operator.
Its matrix elements have a Wigner-Eckart type structure:

```math
\langle \alpha' m' | T^{(\gamma)}_\mu | \alpha m \rangle
  =
  \sum_{\tau}
  C^{\alpha' m'}_{\alpha m, \gamma \mu; \tau}
  \langle \alpha' || T^{(\gamma)} || \alpha \rangle_{\tau}.
```

Here ``C`` is a Clebsch-Gordan coefficient, and ``\tau`` labels outer multiplicities when the same irreducible representation appears more than once.
The reduced matrix element ``\langle \alpha' || T^{(\gamma)} || \alpha \rangle_{\tau}`` contains the part that is not fixed by symmetry alone.

## Relation To SU(2)

For SU(2), irreducible representations are labeled by spin ``S`` and have dimension ``2S + 1``.
The SU(2) Dynkin label ``[a]`` is related to spin by

```math
a = 2S.
```

The familiar angular-momentum addition rule

```math
S_1 \otimes S_2
  =
  |S_1 - S_2| \oplus \cdots \oplus (S_1 + S_2)
```

is the simplest example of tensor-product decomposition.
For SU(Nc), the rank is ``N_c - 1``, so representation labels, multiplicities, and Clebsch-Gordan structures become richer.

## Notation Summary

- ``N_c`` is the dimension of the fundamental representation, not the rank of SU(Nc).
- ``\alpha, \beta, \gamma`` denote irreducible SU(Nc) representations.
- ``\alpha \otimes \beta`` denotes a tensor product of representations.
- ``N_{\alpha\beta}^{\gamma}`` is a Littlewood-Richardson coefficient.
- ``\alpha^\ast`` denotes the conjugate representation of ``\alpha``.
- ``\mathbf{1}`` denotes the trivial representation, also called a singlet.

