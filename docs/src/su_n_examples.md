# Examples of SU(Nc) Representations

This page collects small SU(Nc) examples that make the notation in [SU(Nc) Representation Theory](representation_theory.md) more concrete.
It focuses on representation labels and tensor-product decompositions, not on algorithms.

## Reading The Labels

An SU(Nc) irreducible representation can be written in several equivalent ways.
The three most common forms are:

- a Young diagram, described here by row lengths such as ``(2,1)``;
- a Dynkin label such as ``[1,1,0]``;
- a dimension name such as ``\mathbf{3}``, ``\mathbf{8}``, or ``\mathbf{15}`` when the context is clear.

Dimension names are convenient but not unique across different groups.
For example, ``\mathbf{6}`` in SU(3) and ``\mathbf{6}`` in SU(4) refer to different representation-theoretic objects.
Dynkin labels and Young-diagram row lengths are usually safer.

## SU(2)

SU(2) has rank one, so each irreducible representation has a single Dynkin label ``[a]``.
It corresponds to spin

```math
S = \frac{a}{2},
```

and has dimension

```math
\dim[a] = a + 1 = 2S + 1.
```

Some common SU(2) representations are:

| Spin ``S`` | Dynkin label | Dimension |
|:----------:|:------------:|:---------:|
| ``0`` | ``[0]`` | ``1`` |
| ``1/2`` | ``[1]`` | ``2`` |
| ``1`` | ``[2]`` | ``3`` |
| ``3/2`` | ``[3]`` | ``4`` |

The familiar tensor-product rule is

```math
S_1 \otimes S_2
  =
  |S_1 - S_2| \oplus \cdots \oplus (S_1 + S_2).
```

For example,

```math
\mathbf{2} \otimes \mathbf{2}
  =
  \mathbf{1} \oplus \mathbf{3}.
```

In Dynkin labels, this is

```math
[1] \otimes [1] = [0] \oplus [2].
```

## SU(3)

SU(3) has rank two, so irreducible representations are labeled by ``[p,q]``.
The corresponding Young-diagram row lengths are

```math
(\lambda_1,\lambda_2) = (p + q, q).
```

Some common SU(3) representations are:

| Name | Dynkin label | Row lengths | Dimension |
|:-----|:-------------|:------------|----------:|
| singlet | ``[0,0]`` | ``()`` | ``1`` |
| fundamental | ``[1,0]`` | ``(1)`` | ``3`` |
| antifundamental | ``[0,1]`` | ``(1,1)`` | ``3`` |
| adjoint | ``[1,1]`` | ``(2,1)`` | ``8`` |
| two-index symmetric | ``[2,0]`` | ``(2)`` | ``6`` |
| two-index antisymmetric | ``[0,1]`` | ``(1,1)`` | ``3`` |

The most useful small decompositions are:

```math
\mathbf{3} \otimes \mathbf{3}
  =
  \mathbf{6} \oplus \bar{\mathbf{3}},
```

```math
\mathbf{3} \otimes \bar{\mathbf{3}}
  =
  \mathbf{1} \oplus \mathbf{8}.
```

In Dynkin labels,

```math
[1,0] \otimes [1,0]
  =
  [2,0] \oplus [0,1],
```

```math
[1,0] \otimes [0,1]
  =
  [0,0] \oplus [1,1].
```

The second decomposition shows explicitly how a fundamental and an antifundamental can form a singlet.

## SU(4)

SU(4) has rank three, so irreducible representations are labeled by ``[a,b,c]``.
The corresponding Young-diagram row lengths are

```math
(\lambda_1,\lambda_2,\lambda_3)
  =
  (a + b + c, b + c, c).
```

Some common SU(4) representations are:

| Name | Dynkin label | Row lengths | Dimension |
|:-----|:-------------|:------------|----------:|
| singlet | ``[0,0,0]`` | ``()`` | ``1`` |
| fundamental | ``[1,0,0]`` | ``(1)`` | ``4`` |
| antifundamental | ``[0,0,1]`` | ``(1,1,1)`` | ``4`` |
| two-index antisymmetric | ``[0,1,0]`` | ``(1,1)`` | ``6`` |
| two-index symmetric | ``[2,0,0]`` | ``(2)`` | ``10`` |
| adjoint | ``[1,0,1]`` | ``(2,1,1)`` | ``15`` |

The fundamental tensor products include:

```math
\mathbf{4} \otimes \mathbf{4}
  =
  \mathbf{10} \oplus \mathbf{6},
```

```math
\mathbf{4} \otimes \bar{\mathbf{4}}
  =
  \mathbf{1} \oplus \mathbf{15}.
```

In Dynkin labels,

```math
[1,0,0] \otimes [1,0,0]
  =
  [2,0,0] \oplus [0,1,0],
```

```math
[1,0,0] \otimes [0,0,1]
  =
  [0,0,0] \oplus [1,0,1].
```

## Adjoint Representation

For SU(Nc), the adjoint representation has dimension

```math
N_c^2 - 1.
```

Its Dynkin label is

```math
[1,0,\ldots,0,1],
```

for ``N_c > 2``.
For SU(2), the adjoint representation is the spin-1 representation ``[2]``.

The adjoint representation appears in the decomposition

```math
\mathbf{N_c} \otimes \bar{\mathbf{N_c}}
  =
  \mathbf{1} \oplus \mathrm{Adj}.
```

This identity is one of the most common ways to recognize the singlet and adjoint sectors.

## Conjugation Examples

Conjugation reverses Dynkin labels:

```math
[a_1,a_2,\ldots,a_{N_c-1}]^\ast
  =
  [a_{N_c-1},\ldots,a_2,a_1].
```

Examples:

| Group | Representation | Conjugate |
|:------|:---------------|:----------|
| SU(2) | ``[a]`` | ``[a]`` |
| SU(3) | ``[1,0]`` | ``[0,1]`` |
| SU(3) | ``[2,0]`` | ``[0,2]`` |
| SU(4) | ``[1,0,0]`` | ``[0,0,1]`` |
| SU(4) | ``[0,1,0]`` | ``[0,1,0]`` |

The SU(4) two-index antisymmetric representation ``[0,1,0]`` is self-conjugate.

## Practical Caution

Dimension labels are compact, but they can hide important distinctions.
When writing implementation notes or checking decompositions, prefer Dynkin labels or row lengths unless the group and convention are completely clear.

