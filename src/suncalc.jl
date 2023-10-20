"""
irrep(Nc, index)
retrieves an irrep from an index
"""
function irrep(Nc, index)
    elem = zeros(Int, Nc)
    i = 0
    while index > 0 && i < Nc
        j = 1
        while(binomial(Nc - i - 1 + j, Nc - i - 1) <= index)
            elem[i + 1] = j
            j <<= 1
        end
        j = elem[i + 1] >> 1
        while j > 0
            if binomial(Nc - i - 1 + (elem[i + 1] | j), Nc - i - 1) <= index
                elem[i + 1] |= j
            end
            j >>= 1
        end
        index -= binomial(Nc - i - 1 + elem[i + 1], Nc - i - 1)
        elem[i + 1] += 1
        i += 1
    end
    elem
end

"""
irreplist(Nc, widthmax)
generates an irrep list
"""
function irreplist(Nc, widthmax)
    index = 0
    rtn = SUNIrrep{Nc}[]
    elem = irrep(Nc, index)
    while elem[1] <= widthmax
        push!(rtn, SUNIrrep(Tuple(elem)))
        index += 1
        elem .= irrep(Nc, index)
    end
    rtn
end

"""
mult = outer_multiplicity(α, β, γ)
The outer multiplicity of γ in α ⊗ β.
"""
function outer_multiplicity(α, β, γ)
    get(directproduct(α, β), γ, 0)
end

"""
OM = OM_matrix(βlist1, βlist2, γ)
returns outer multiplicity matrix.
"""
function OM_matrix(βlist1, βlist2, γ)
    [outer_multiplicity(βi, βj, γ) for βi in βlist1, βj in βlist2]
end

"""
OM = OM_matrix(βlist, γ)
returns outer multiplicity matrix.
"""
function OM_matrix(βlist, γ)
    OM_matrix(βlist, βlist, γ)
end

"""
trivialirrep(Val(Nc))
returns the trivial irrep
"""
function trivialirrep(val)
    SUNIrrep(ntuple(i -> 0, val))
end

"""
adjointirrep(Val(Nc))
returns the adjoint irrep
"""
function adjointirrep(::Val{Nc}) where Nc
    SUNIrrep(ntuple(i -> 1 + (i == 1) - (i == Nc), Val(Nc)))
end

"""
fundamentalirrep(Val(Nc))
returns the fundamental irrep
"""
function fundamentalirrep(val)
    SUNIrrep(ntuple(i -> 0 + (i == 1), val))
end

"""
wigner3ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν::SUNIrrep{Nc})
Returns the array of Wigner 3ν coefficients in the format, `W[τ′, τ]`.
This is a matrix applied when two representations are exchanged.
This is different from the so-called 3j-symbols or Clebsch–Gordan coefficients.
"""
function wigner3ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν::SUNIrrep{Nc}) where Nc
    ti = trivialirrep(Val(Nc))
    wigner9ν(ti, ν1, ν1, ν2, ti, ν2, ν2, ν1, ν)[1, 1, :, 1, 1, :]
end

"""
wigner6ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν23::SUNIrrep{Nc})
Returns the array of Wigner 6ν coefficients in the format, `W[τ23, τ′, τ12, τ]`.
"""
function wigner6ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν23::SUNIrrep{Nc}) where Nc
    racahU(ν1, ν2, ν, ν3, ν12, ν23)
end

"""
racahU(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν23::SUNIrrep{Nc})
Returns the array of Racah U-coefficients in the format, `U[τ23, τ′, τ12, τ]`.
"""
function racahU(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν23::SUNIrrep{Nc}) where Nc
    if Nc == 2
        μ1, μ2, μ12, μ3, μ, μ23 = map(first ∘ weight, (ν1, ν2, ν12, ν3, ν, ν23))
        j1, j2, j12, j3, j, j23 = (μ1, μ2, μ12, μ3, μ, μ23) .// 2
        τ12max = δ(j1, j2, j12)
        τmax = δ(j12, j3, j)
        τ23max = δ(j2, j3, j23)
        τ′max = δ(j1, j23, j)
        if !all([τ23max, τ′max, τ12max, τmax])
            return zeros(τ23max, τ′max, τ12max, τmax)
        else
            return reshape([convert(Float64, signedroot((μ12 + 1) * (μ23 + 1)) * racahW(j1, j2, j, j3, j12, j23))], 1, 1, 1, 1)
        end
    end
end

"""
wigner9ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν4::SUNIrrep{Nc}, ν34::SUNIrrep{Nc}, ν13::SUNIrrep{Nc}, ν24::SUNIrrep{Nc}, ν::SUNIrrep{Nc})
Returns the array of Wigner 9ν coefficients in the format, `W[τ13, τ24, τ′, τ12, τ34, τ]`.
"""
function wigner9ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν4::SUNIrrep{Nc}, ν34::SUNIrrep{Nc}, ν13::SUNIrrep{Nc}, ν24::SUNIrrep{Nc}, ν::SUNIrrep{Nc}) where Nc
    if Nc == 2
        μ1, μ2, μ12, μ3, μ4, μ34, μ13, μ24, μ = map(first ∘ weight, (ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν))
        j1, j2, j12, j3, j4, j34, j13, j24, j = (μ1, μ2, μ12, μ3, μ4, μ34, μ13, μ24, μ) .// 2
        τ12max = δ(j1, j2, j12)
        τ34max = δ(j3, j4, j34)
        τmax = δ(j12, j34, j)
        τ13max = δ(j1, j3, j13)
        τ24max = δ(j2, j4, j24)
        τ′max = δ(j13, j24, j)
        if !all([τ13max, τ24max, τ′max, τ12max, τ34max, τmax])
            return zeros(τ13max, τ24max, τ′max, τ12max, τ34max, τmax)
        else
            return reshape([sqrt((μ12 + 1) * (μ34 + 1) * (μ13 + 1) * (μ24 + 1)) * wigner9j(j1, j2, j12, j3, j4, j34, j13, j24, j)], 1, 1, 1, 1, 1, 1)
        end
    end
end

"""
wigner9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
Tentative function until WignerSymbols.jl supports 9j.
"""
function wigner9j(j1, j2, j3, j4, j5, j6, j7, j8, j9)
    imax = convert(Int, min(j1 + j9, j2 + j6, j4 + j8) * 2)
    imin = imax % 2
    sumres = 0.0
    for kk in imin : 2 : imax
        sumres += (kk + 1) * racahW(j1, j2, j9, j6, j3, kk // 2) * racahW(j4, j6, j8, j2, j5, kk // 2) * racahW(j1, j4, j9, j8, j7, kk // 2)
    end
    sumres
end

"""
wigner6νrev(ν1, ν2, ν12, ν3, ν, ν23)
computes the 6ν coefficient in the reversed form
"""
function wigner6νrev(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν23::SUNIrrep{Nc}) where Nc
    A = wigner6ν(ν1, ν2, ν12, ν3, ν, ν23)
    a, b, c, d = size(A)
    Array(reshape(reshape(A, a * b, c * d)', c, d, a, b))
end