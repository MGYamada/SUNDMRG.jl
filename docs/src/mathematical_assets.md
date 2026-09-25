# Mathematical Assets and Documentation Plan

SUNDMRG contains reusable representation-theory machinery beneath its DMRG
solver. This page is a starting inventory for documenting that machinery: what
each component represents, where its implementation and checks live, and which
conventions still need a worked explanation. It is not a complete specification
of every internal routine.

The representation implementation is assembled in
`src/representation_theory.jl`. The exported interfaces are described in the
[API Reference](api.md). SYT, sparse-vector, and subduction routines listed here
are internal; the ED utilities in `validation/` are development tools. Listing
them here does not turn them into supported public APIs.

## Reading Order

Start with [SU(Nc) Representation Theory](representation_theory.md),
[Representation Examples](su_n_examples.md), and
[Representation Labels](representation_notation.md). Then follow the
implementation from Young diagrams and tableau indexing to permutation actions,
subduction, recoupling, and coefficient tables. The inventory below supplies
the starting points for the SYT and subduction notes that are still to be written.

[Wigner/Racah Coefficients](wigner_racah.md) already defines the recoupling
symbols, multiplicity-axis order, and six table slots.
[Coefficient Tables](coefficient_tables.md) covers loading and generation.
Use [SUNDMRG Algorithm](algorithm.md) to see how these objects enter the DMRG
calculation; use the repository's `validation/README.md` for the ED design and
execution instructions.

## Implementation Inventory

Paths in this inventory are relative to the repository root. Read the routines
in their `RepresentationTheory` module context rather than including individual
source files as standalone libraries.

| Asset | Implementation | Documentation to develop or connect |
|:------|:---------------|:---------------------------------------|
| SU(N) labels, representation enumeration, and tensor-product multiplicities | `src/suncalc.jl`: `irrep`, `irreplist`, `outer_multiplicity`, `OM_matrix`; representation types and tensor products use `SUNRepresentations` | Extend the label guide with the different meanings of dimension and multiplicity, and the boundary between dependency-provided operations and package helpers. |
| SYT counting and indexing | `src/sytx.jl`: `multiplicity`, `SYTdiagram`, `bf`, `subdiagram` | Define the graph arrays and path ordering; explain tableau-to-index traversal and the role of subdiagrams with a small worked example. |
| Permutation action in a tableau basis | `src/sytx.jl`: `P!`, `Papply!`, `Papply2!` | Specify adjacent exchanges, composition order, full-space versus subdiagram coordinates, and each routine's mutation contract. |
| Sparse coefficient arithmetic | `src/sparsevec2.jl`: `SparseVector2`, `sparsevec2`, addition, scaling, inner products | Explain sorted sparse coordinates and amplitude pruning. Distinguish this numerical representation from an exact linear map. |
| Subduction and multiplicity-basis choice | `src/subduction.jl`: `representatives`, `antisymmetrize`, `SDC`, `_SDC`, `gaugefix!` and its helpers | Document embeddings, normalization, axes, gauge fixing, tolerance choices, and the `perm`, `f1`, and `f4` options. Preserve the attribution attached to the gauge-fixing helpers. |
| Exchange and recoupling coefficients | `src/tablecalc.jl`: `_3ν`, `_6ν`, `_6νrev`, `_9ν`; wrappers in `src/suncalc.jl` | Build on the existing Wigner/Racah definitions. Add small examples that make phases, axis order, forbidden couplings, and the SU(2) specialization explicit. |
| Versioned coefficient tables | `src/table3nu.jl`, `src/table4.jl`, `src/table.jl`, and `table_9ν` in `src/tablecalc.jl`; bundled data in `jld2/` | Connect the existing six-slot map to generation provenance, sampled coefficient checks, and the consuming DMRG operations. Keep generation/lifecycle instructions in the coefficient-table guide. |
| Symmetry-resolved ED and independent references | `validation/sun_ed.jl`, `validation/sun_ed_krylov.jl`, and `test/reference_sun.jl` | Explain how shared SYT machinery produces an untruncated reference, which checks are independent, and what residuals and degenerate-subspace comparisons establish. |

## Conventions to Make Explicit

The first notes should resolve the following questions before adding larger
examples or derivations:

- Which object is being labelled: a normalized SU(N) irrep or an L-box Young
  diagram? The ED keeps all L boxes, including full columns of height N; its
  singlet shape is rectangular. The package-facing singlet irrep label is zero.
- Which size is being counted: SU(N) irrep dimension, number of tableaux,
  tensor-product outer multiplicity, or retained DMRG multiplets? The function
  `multiplicity` in `sytx.jl` counts tableaux; `outer_multiplicity` answers a
  different question about a tensor product.
- What is the ordering of sites, graph paths, tensor factors, and multiplicity
  axes? How does an index in a subdiagram relate to the full tableau space?
- What basis, phase/gauge convention, normalization, and numerical tolerance
  enter the output? Which comparisons remain meaningful when a multiplicity or
  energy eigenspace has dimension greater than one?
- Does an operation mutate inputs, allocate new storage, or prune amplitudes?
  The ED extracts adjacent-exchange transitions from the SYT routines, then
  applies them to dense vectors without sparse-addition pruning.

The energy/correlation conversion is another explicit boundary: SU(2) outputs
use the spin convention, while the current N > 2 reported energy uses the
permutation convention and the bond correlation is centered. Keep worked
comparisons consistent with [SUNDMRG Algorithm](algorithm.md), identical bond
multiplicities, and the conversion formulas in `validation/README.md`.

## Validation Map and Its Limits

The following checks exist in the repository. This map describes their scope;
it does not report a new test run or establish a theorem for untested sizes.

| Evidence | Existing checks | What it supports and what it does not |
|:---------|:----------------|:--------------------------------------|
| Checks against mathematical definitions | `test/test_representation_theory_definitions.jl`: enumeration, hook-length tableau counts, Weyl dimensions, and tensor-product dimension sums | Selected inputs agree with the stated definitions. Two implementations of the same formula can still share a mistaken convention. |
| Algebraic identities in the SYT basis | `test/test_sun_ed.jl`: transposition squares, braid/commutation relations, and the complete-graph central element | Small representations satisfy the tested identities; this is finite numerical evidence. |
| Independent product-basis reference | `test/reference_sun.jl` and `test/test_sun_ed.jl`: direct color swaps, all-sector spectra, and Casimir-selected singlets | Tests the small-system SYT Hamiltonian against a construction that does not use SYT indexing or recoupling. Equal color populations alone do not select singlets. |
| Subduction normalization | `test/test_representation_theory_definitions.jl`: SU(3) adjoint-product multiplicity and SDC orthonormality; small cases in `test/test_representation_theory_internal.jl` | Checks the selected multiplicity spaces. Orthonormality alone does not determine the gauge or validate every coefficient phase. |
| Recoupling and stored-table consistency | `test/test_tables_small.jl`, `test/test_tables_ground_truth.jl` | SU(2) cases and sampled SU(3)–SU(5) table entries are checked. Regenerating entries through shared SDC routines is a consistency check, not a fully independent oracle. |
| Physical DMRG comparison | `test/test_sun_ed_dmrg.jl`: untruncated six-site SU(3) ground/excited energies and ground correlations | Compares DMRG against the small singlet reference across both solver and storage modes. It does not establish accuracy after truncation at arbitrary sizes. |
| Iterative ED convergence | `validation/runtests.jl`: direct residuals, restarted solves, seed changes, and a degenerate projector | Checks numerical convergence and the tested subspaces. Random starts and small residuals alone do not prove spectral completeness. |

Separate mathematical statements from the evidence used to check an
implementation. A future note should cite the definition or derivation, name
the relevant test, and state its parameter range and tolerance. Keep actual run
results and unavailable configurations in `ROADMAP.md`; preserve explicit gaps
when independent validation is not available.

## Outline for Each Mathematical Note

Use the same small set of questions for each component:

1. **Object and scope:** define the mathematical map, admissible inputs, and a
   reference for the definition or derivation.
2. **Representation and conventions:** define labels, array axes, ordering,
   normalization, phase/gauge choices, and forbidden or empty cases.
3. **Implementation contract:** identify the responsible functions, their
   inputs/outputs, mutation, numerical approximations, and cost or size limits.
4. **Worked example:** choose a small existing fixture, show intermediate
   mathematical objects, and check the expected output. Prefer an executable
   documentation example where its runtime and dependencies are suitable.
5. **Validation and limitations:** identify independent references separately
   from shared-code comparisons, give tolerances, and list unverified claims.
6. **Consumers and reuse:** explain how the result enters subduction,
   recoupling, tables, ED, or DMRG, and whether the interface is public or internal.

The first writing pass is the label/dimension clarification and a SYT/indexing
note, followed by permutation/sparse-arithmetic contracts and a concise
subduction note. Detailed recoupling derivations and an extensive example
catalog can follow. The preparation itself changes neither algorithms nor the
coefficient-table format and does not require generating new tables.
