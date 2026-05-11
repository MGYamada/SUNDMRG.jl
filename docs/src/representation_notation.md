# Representation Labels In SUNDMRG.jl

SUNDMRG.jl follows the Young row-length convention for SU(Nc) representation labels.
When labels appear in package-facing contexts, trailing zeros may be retained so that the tuple length is ``N_c``.

For example, in SU(3):

| Representation | Mathematical row lengths | Package-facing label |
|:---------------|:-------------------------|:---------------------|
| singlet | ``()`` | ``(0, 0, 0)`` |
| fundamental | ``(1)`` | ``(1, 0, 0)`` |
| antifundamental | ``(1, 1)`` | ``(1, 1, 0)`` |
| adjoint | ``(2, 1)`` | ``(2, 1, 0)`` |

Dynkin labels are obtained from adjacent differences:

```math
a_i = \lambda_i - \lambda_{i+1}.
```

Thus the SU(3) label ``(2,1,0)`` corresponds to Dynkin label ``[1,1]``.

