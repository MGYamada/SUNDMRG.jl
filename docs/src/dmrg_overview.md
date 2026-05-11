# DMRG Overview

The density matrix renormalization group (DMRG) is a variational method for finding low-energy states of quantum many-body systems.
It is especially effective when the target state has limited entanglement across spatial cuts.

This page gives a high-level overview of the method.
It avoids package-specific input syntax and focuses on the general algorithmic picture.

## Variational Idea

The exact Hilbert space of a many-body system grows exponentially with system size.
DMRG works by keeping only the most relevant states for describing the target wavefunction across a bipartition.

The central approximation is controlled by a bond dimension ``m``.
Larger ``m`` allows more entanglement and usually gives more accurate results, but increases memory use and computational cost.

At a cut dividing the system into left and right parts, a wavefunction can be written in Schmidt form:

```math
|\psi\rangle
  =
  \sum_i s_i\, |i\rangle_L \otimes |i\rangle_R.
```

The Schmidt values ``s_i`` quantify the entanglement across the cut.
DMRG keeps the most important Schmidt sectors and discards the rest.

## Reduced Density Matrix

The name DMRG comes from the reduced density matrix used to choose the truncated basis.
Given a normalized wavefunction ``|\psi\rangle`` for a bipartition, the reduced density matrix of the left subsystem is

```math
\rho_L = \mathrm{Tr}_R |\psi\rangle\langle\psi|.
```

Its eigenvalues measure how much weight each left-subsystem state carries in the full wavefunction.
Keeping the eigenvectors with the largest eigenvalues gives an optimal truncated basis for that bipartition in the least-squares sense.

The discarded weight is often used as a truncation error estimate:

```math
\epsilon = \sum_{i > m} \lambda_i,
```

where ``\lambda_i`` are the density-matrix eigenvalues sorted in descending order.

## Blocks And Superblocks

Traditional finite-system DMRG describes the system using blocks.
A typical step builds a superblock from a system block, one or more active sites, and an environment block.
The target state of this superblock is obtained by solving an effective eigenvalue problem.

Conceptually, one step does the following:

1. Build an effective Hamiltonian for the current superblock.
2. Solve for the target state, usually the ground state or a low-lying state.
3. Form a reduced density matrix for the side being enlarged.
4. Diagonalize the reduced density matrix.
5. Keep the most important ``m`` states.
6. Transform operators into the new truncated basis.

The result is a new block that approximates a larger part of the system.

## Warmup And Sweeps

Finite-system DMRG is usually organized into two stages.

During the warmup stage, the algorithm grows blocks until the desired system size is reached.
The environment may be approximate during this stage.

During the sweep stage, the system size is fixed.
The algorithm moves the active cut back and forth through the system, alternately improving the left and right block bases.
Each sweep refines the variational state using a better environment.

Sweeps continue until observables, energy, or entanglement quantities converge to the desired tolerance.

## Entanglement Entropy

The Schmidt values also define the entanglement entropy across a cut.
If ``\lambda_i = s_i^2`` are the eigenvalues of the reduced density matrix, the von Neumann entanglement entropy is

```math
S = -\sum_i \lambda_i \log \lambda_i.
```

Entanglement entropy is useful both as a physical observable and as a diagnostic of how difficult a calculation is.
Large entanglement generally requires a larger bond dimension.

## Symmetry Sectors

When a Hamiltonian has a symmetry, the Hilbert space decomposes into symmetry sectors.
DMRG can exploit this by organizing basis states and operators according to conserved quantum numbers or irreducible representations.

In a symmetry-adapted calculation, the reduced density matrix is block diagonal in symmetry sectors.
The truncation then keeps important states across sectors while preserving the symmetry structure.

For non-Abelian symmetries such as SU(Nc), states are grouped by irreducible representations rather than only by Abelian charges.
The representation-theory pages in this manual describe the notation used for those labels.

## Accuracy Controls

The most important numerical controls in a DMRG calculation are:

- the bond dimension ``m``;
- the truncation error or discarded weight;
- the number of sweeps;
- convergence thresholds for energy and observables;
- the quality of the effective eigensolver.

Increasing ``m`` is the most direct way to improve the variational space.
However, a larger ``m`` also increases the cost of tensor contractions, operator transformations, and effective Hamiltonian applications.

## MPS Perspective

Modern presentations often describe DMRG as a variational optimization over matrix product states (MPS).
In that language, a sweep optimizes local tensors while keeping the rest of the MPS fixed.

Traditional block DMRG and MPS DMRG are closely related views of the same variational principle.
The block language emphasizes renormalized bases and effective operators, while the MPS language emphasizes tensor networks and canonical forms.

Both perspectives are useful.
The rest of this documentation will use whichever viewpoint makes the relevant part of SUNDMRG.jl easiest to explain.

