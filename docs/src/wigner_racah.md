# Wigner/Racah Coefficients

This page connects the recoupling notation used by SUNDMRG.jl with the
implementation functions and the coefficient tables consumed by DMRG runs.

SUNDMRG labels SU(Nc) irreducible representations by Young row lengths, stored as
`SUNIrrep{Nc}` values. In the formulas below, ``\nu_i`` denotes such an
irreducible representation, and ``\tau`` labels an outer-multiplicity channel.
The symbols ``f``, ``a``, and ``1`` denote the fundamental, adjoint, and trivial
representations, respectively.

## Recoupling Symbols

The low-level construction starts from subduction coefficients. If
``C_{\tau}(\nu_1,\nu_2;\nu)`` denotes the embedding of the ``\tau``-th copy of
``\nu`` inside ``\nu_1 \otimes \nu_2``, then the Wigner/Racah symbols used here
are overlaps between two different coupling schemes.

The underscored routines `_3ν`, `_6ν`, and `_9ν` are the general SDC-based
builders used during table generation. The `wigner3ν`, `wigner6ν`, and
`wigner9ν` wrappers provide the SU(2) on-the-fly path used by DMRG runs with
`Nc == 2`; runs with `Nc > 2` consume precomputed table entries instead.

### 3ν

The 3ν symbol exchanges two coupled representations:

```math
[3\nu(\nu_1,\nu_2;\nu)]_{\tau',\tau}
=
\left\langle
  C_{\tau'}(\nu_2,\nu_1;\nu),
  P_{12} C_{\tau}(\nu_1,\nu_2;\nu)
\right\rangle .
```

Implementation:

- `SUNDMRG.RepresentationTheory._3ν(ν1, ν2, ν)`
- `SUNDMRG.RepresentationTheory.wigner3ν(ν1, ν2, ν)`

The returned array has indices `W[τ′, τ]`. This is an exchange matrix on the
outer-multiplicity space. It is not a Wigner 3j symbol and not a Clebsch-Gordan
coefficient.

### 6ν / Racah U

The 6ν symbol changes the parenthesization of three representations:

```math
\left[
\begin{matrix}
\nu_1 & \nu_2 & \nu_{12} \\
\nu_3 & \nu     & \nu_{23}
\end{matrix}
\right]_{\tau_{23},\tau',\tau_{12},\tau}
=
\left\langle
  [\nu_1 \otimes (\nu_2 \otimes \nu_3)_{\nu_{23}}]_{\nu,\tau'},
  [(\nu_1 \otimes \nu_2)_{\nu_{12}} \otimes \nu_3]_{\nu,\tau}
\right\rangle .
```

Implementation:

- `SUNDMRG.RepresentationTheory._6ν(ν1, ν2, ν12, ν3, ν, ν23)`
- `SUNDMRG.RepresentationTheory.wigner6ν(ν1, ν2, ν12, ν3, ν, ν23)`
- `SUNDMRG.RepresentationTheory.racahU(ν1, ν2, ν, ν3, ν12, ν23)`

The returned array has indices `W[τ23, τ′, τ12, τ]`. For SU(2),
`wigner6ν` is implemented as a Racah-U coefficient. If the SU(2) row-length
label is ``\mu = 2j``, then the implementation evaluates a dimension factor
times the usual Racah ``W(j_1,j_2,j,j_3,j_{12},j_{23})``.

### 9ν

The 9ν symbol changes the pairing of four representations:

```math
\left[
\begin{matrix}
\nu_1  & \nu_2  & \nu_{12} \\
\nu_3  & \nu_4  & \nu_{34} \\
\nu_{13} & \nu_{24} & \nu
\end{matrix}
\right]_{\tau_{13},\tau_{24},\tau',\tau_{12},\tau_{34},\tau}
=
\left\langle
  [(\nu_1 \otimes \nu_3)_{\nu_{13}}
   \otimes
   (\nu_2 \otimes \nu_4)_{\nu_{24}}]_{\nu,\tau'},
  P_{23}
  [(\nu_1 \otimes \nu_2)_{\nu_{12}}
   \otimes
   (\nu_3 \otimes \nu_4)_{\nu_{34}}]_{\nu,\tau}
\right\rangle .
```

Here ``P_{23}`` is the formal exchange of the second and third tensor factors,
turning the ordering ``\nu_1 \otimes \nu_2 \otimes \nu_3 \otimes \nu_4`` into
``\nu_1 \otimes \nu_3 \otimes \nu_2 \otimes \nu_4`` before the overlap is taken.
In the implementation this is the `perm = true` branch of the final
`SDC(ν12, ν34, ν, ...; f1 = N1, f4 = N4)` call.

Implementation:

- `SUNDMRG.RepresentationTheory._9ν(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)`
- `SUNDMRG.RepresentationTheory.wigner9ν(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)`

The returned array has indices `W[τ13, τ24, τ′, τ12, τ34, τ]`. For SU(2),
`wigner9ν` is evaluated from the ordinary Wigner 9j symbol with the corresponding
dimension factor. The public `wigner9ν` wrapper is implemented only for SU(2);
general SU(Nc) 9ν construction is available through `_9ν` and the table
generators. DMRG runs with `Nc > 2` use precomputed tables.

## SU(2) Correspondence

For SU(2), the package row-length label ``\mu`` corresponds to spin ``j`` by

```math
\mu = 2j.
```

Thus the SU(2) fundamental representation is ``j = 1/2``, the adjoint
representation is ``j = 1``, and the trivial representation is ``j = 0``.

| SUNDMRG object | SU(2) angular-momentum object | Implementation |
|:---------------|:------------------------------|:---------------|
| `wigner3ν(ν1, ν2, ν)` | Exchange matrix between ``ν_1 \otimes ν_2`` and ``ν_2 \otimes ν_1`` coupling bases | `wigner9ν(ti, ν1, ν1, ν2, ti, ν2, ν2, ν1, ν)[1, 1, :, 1, 1, :]`, where `ti` is trivial |
| `wigner6ν(ν1, ν2, ν12, ν3, ν, ν23)` | Racah-U coefficient | `racahU(...)` |
| `wigner9ν(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)` | Wigner 9j with dimension normalization | `sqrt((μ12 + 1)(μ34 + 1)(μ13 + 1)(μ24 + 1)) * wigner9j(...)` |

The SU(2) wrappers return arrays whose multiplicity dimensions are `1` for
allowed couplings and `0` for forbidden couplings. The array shape is kept
compatible with the SU(Nc) code path.

## Coefficient Generation

The table-generation pipeline separates expensive coefficient construction from
DMRG execution:

| Stage | Output file | Mathematical content | Implementation |
|:------|:------------|:---------------------|:---------------|
| `make_table3nu(Nc, widthmax)` | `table3nuhalf_SU$(Nc)_$(widthmax).jld2` | Selected half-table of 3ν exchange matrices for allowed sectors | `_3ν` |
| `make_table4(Nc, widthmax)` | `table4half_SU$(Nc)_$(widthmax).jld2` | Selected half-table of 9ν slices for adjoint-adjoint interactions | `_9ν`-style SDC overlaps |
| `make_table(Nc, widthmax)` | `table_SU$(Nc)_$(widthmax).jld2` | The six-table tuple consumed by DMRG | `table_9ν` |

`make_table` also fills symmetry-related entries and derives the specialized
tables needed by the DMRG kernels. The final `tables` object is therefore a tuple
of six coefficient dictionaries rather than a raw dump of all 3ν, 6ν, and 9ν
symbols.

## DMRG Table Map

The following table gives the one-to-one correspondence between each table slot,
the mathematical coefficient it stores, the on-the-fly SU(2) helper, and the DMRG
operation that consumes it.

| Table slot | Key | Stored coefficient | SU(2) helper | DMRG use |
|:-----------|:----|:-------------------|:-------------|:---------|
| `tables[1]` | `(α1, β1, α2, β2)` | ``9ν(α_1,f,β_1; a,1,a; α_2,f,β_2)[:,1,1,1,1,:]`` | `on_the_fly_calc1` | Recouple an existing block tensor operator after adding a fundamental site. Used to build `BlockEnlarging`, then reused in Lanczos, measurements, and density-matrix mixing. |
| `tables[2]` | `(α, β1, β2)` | ``6ν(α,f,β_1; a,β_2,f)[1,1,1,:]`` | `on_the_fly_calc2` | Build the spin tensor for the newly added site in `_build_spin_tensor`. |
| `tables[3]` | `(α1, β, α2)` | ``9ν(α_1,f,β; a,a,1; α_2,f,β)[:,1,1,1,1,1]`` | `on_the_fly_calc3` | Add intra-block bond interactions to the enlarged block Hamiltonian. |
| `tables[4]` | `(α1, β1, γ, α2, β2)` | ``9ν(α_1,β_1,γ; a,a,1; α_2,β_2,γ)[:,:,:,:,1,1]`` | `on_the_fly_calc4` | Build the superblock inter-block interaction term `H2`. |
| `tables[5]` | `(αj, βl, αi, γ, βk)` | ``6ν(α_j,f,β_l; α_i,γ,β_k)[1,:,1,:] \, 3ν(α_i,f,β_k)[1,1]`` | `on_the_fly_calc5` | Transform the optimized wavefunction into the next-step initial guess in `eig_prediction`. |
| `tables[6]` | `(α, β, γ)` | ``3ν(α,β,γ)`` | `on_the_fly_calc6` | Reverse the system/environment ordering of a wavefunction in `wavefunction_reverse`. |

The letters ``α`` and ``β`` follow the local variable names used by each kernel,
not one global convention across all six tables. In rows involving a one-site
enlargement, ``α`` labels sectors before adding the fundamental site and ``β``
labels sectors after the addition. In the superblock and wavefunction-transfer
rows, the subscripts instead distinguish the coupled block sectors before and
after adjoint or fundamental recoupling. The representation ``γ`` is the total
sector carried by the relevant system-environment product.

## On-The-Fly And Precomputed Modes

At runtime, SU(2) runs set `on_the_fly = true`. Each DMRG kernel calls
`on_the_fly_calc1` through `on_the_fly_calc6`, which evaluate the small Wigner or
Racah coefficients as needed.

For `Nc > 2`, `on_the_fly = false`. The user must provide both `widthmax` and the
precomputed `tables` tuple. This keeps the DMRG run focused on linear algebra and
table lookup, while the expensive SDC-based coefficient construction is performed
ahead of time.
