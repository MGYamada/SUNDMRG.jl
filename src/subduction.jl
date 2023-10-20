#=
    subduction coefficients routines
    Copyright (C) 2021 Masahiko G. Yamada <myspinor@gmail.com>,
    except where otherwise indicated.

    The gaugefix!(), qrpos!(), cref!(), and findabsmax() functions are:

    MIT License

    Copyright (c) 2020 maartenvd

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
=#

const tol_nullspace = 1e-13

"""
reps = representatives(S1, columns, γ)
returns representative SYTx.
"""
function representatives(S1, columns, γ)
    if isempty(columns)
        return [Tuple(vcat(S1...))]
    end
    N = sum(γ)
    Nc = length(γ)
    while length(S1) < Nc
        push!(S1, Int[])
    end
    reps = NTuple{N, Int}[]
    L = length(columns[1])
    x = (1 << L) - 1
    for i in 1 : binomial(Nc, L)
        St = deepcopy(S1)
        k = 1
        for j in 1 : Nc
            if (x >> (j - 1)) & 1 == 1
                push!(St[j], columns[1][k])
                k += 1
            end
        end
        α = length.(St)
        if α == sort(α, rev = true) && all(α .<= γ)
            append!(reps, representatives(St, columns[2 : end], γ))
        end
        s = x & (-x)
        r = s + x
        x = r | ((xor(x, r) >> 2) ÷ s)
    end
    reps
end

"""
φs = antisymmetrize(reps, columns, γ, initj, dim, N1, N, E, D, F, T)
antisymmetrizes the representative.
"""
function antisymmetrize(reps, columns, γ, initj, dim, N1, N, E, D, F, T)
    len = length(reps)
    rtn = map(reps) do rep
        St = collect(rep)
        j = initj
        t = zero(T)
        for i in N1 : N - 1
            position = findfirst(isequal(i + 1), St)
            k = findfirst(isequal(position), D[i][j])
            t += F[i][j][k]
            j = E[i][j][k]
        end
        sparsevec2([t + 1], [1.0], dim)
    end
    for column in columns
        c = length(column)
        rtnnew = [spzeros2(Float64, T, dim) for i in 1 : len]
        for i in 1 : factorial(c)
            p = Permutation(c, i)
            pp = CoxeterDecomposition(p)
            temp = deepcopy(rtn)
            for term in reverse(pp.terms)
                for k in 1 : len
                    temp[k] = Papply!(temp[k], column[term], γ, initj, N1, N, E, D, F, T)
                end
            end
            for k in 1 : len
                lmul!(sign(p), temp[k])
                rtnnew[k] += temp[k]
            end
        end
        for k in 1 : len
            rtn[k] = rtnnew[k]
        end
    end
    for k in 1 : len
        lmul!(1.0 / norm(rtn[k]), rtn[k])
    end
    rtn
end

"""
coeff = SDC(ν1, ν2, ν, vec1, vec2; perm = false, f1 = 0, f4 = 0)[:, τ]
computes the subduction coefficient.
"""
function SDC(μ1::NTuple{Nc, Int}, μ2::NTuple{Nc, Int}, μ::NTuple{Nc, Int}, vec1::SparseVector2{Float64, T}, vec2::SparseVector2{Float64, T}; perm = false, f1 = 0, f4 = 0) where {Nc, T}
    mult = T(multiplicity(μ))
    N1 = sum(μ1)
    N2 = sum(μ2)
    N = sum(μ)
    if N1 == 0
        rtn = vec1[1] * vec2
        return [rtn]
    elseif N2 == 0
        rtn = vec1 * vec2[1]
        return [rtn]
    end
    ν1 = filter(x -> x > 0, μ1)
    ν2 = filter(x -> x > 0, μ2)
    ν = filter(x -> x > 0, μ)
    L1 = length(ν1)
    L2 = length(ν2)
    L3 = length(ν)

    V, E, D = SYTdiagram(ν)
    B, F = bf(N, V, E, T)
    φnew, dim, initj = _SDC(ν1, ν2, ν, E, D, B, F, T)
    OM = length(φnew)

    V1, E1, D1 = SYTdiagram(ν1)
    B1, F1 = bf(N1, V1, E1, T)
    V2, E2, D2 = SYTdiagram(ν2)
    B2, F2 = bf(N2, V2, E2, T)

    φsum = [spzeros2(Float64, T, dim) for k in 1 : OM]
    cumν2 = vcat(0, cumsum(ν2)...)
    ps = Permutation[]
    for l2 in vec2.nzind
        r = l2 - 1
        j = 1
        Sl2 = ones(Int, N2)
        for i in 1 : N2 - 1
            k = searchsortedlast(F2[i][j], r)
            Sl2[D2[i][j][k]] = i + 1
            r -= F2[i][j][k]
            j = E2[i][j][k]
        end

        vertical = Int[]
        for i in 1 : ν2[1]
            j = 1
            while j <= L2 && i <= ν2[j]
                push!(vertical, Sl2[cumν2[j] + i])
                j += 1
            end
        end
        push!(ps, Permutation(vertical))
    end

    temp = zeros(Int, L2)
    S2 = map(α -> zeros(Int, α), collect(ν2))
    j = 1
    for i in 1 : N2
        temp[j] += 1
        S2[j][temp[j]] = i
        if j == length(temp) || temp[j + 1] == ν2[j + 1]
            j = 1
        else
            j += 1
        end
    end
    St = vcat(S2...)
    q = Permutation(N2)
    inds = copy(vec2.nzind)

    while !isempty(ps)
        pps = map(p -> CoxeterDecomposition(p * q'), ps)
        i2 = argmin(map(x -> length(x.terms), pps))
        pp = pps[i2]

        l2 = inds[i2]
        for term in reverse(pp.terms)
            m = findfirst(isequal(term), St)
            m1 = 0
            m2 = 0
            temp = m
            while temp > 0
                m1 += 1
                m2 = temp
                temp -= ν2[m1]
            end
            n = findfirst(isequal(term + 1), St)
            n1 = 0
            n2 = 0
            temp = n
            while temp > 0
                n1 += 1
                n2 = temp
                temp -= ν2[n1]
            end
            axial = Float64((n1 - m1) - (n2 - m2))
            ρ = 1.0 / axial
            factor = ρ / sqrt(1 - ρ ^ 2)
            for k in 1 : OM
                φnew[k] = P!(factor, axial, φnew[k], term + N1, ν, initj, N1, N, E, D, F, T)
            end
            St[m], St[n] = St[n], St[m]
        end
        for k in 1 : OM
            φsum[k] += vec2[l2] * φnew[k]
        end

        q = popat!(ps, i2)
        popat!(pps, i2)
        popat!(inds, i2)
    end

    I = [T[] for k in 1 : OM]
    V = [Float64[] for k in 1 : OM]
    cumν1 = vcat(0, cumsum(ν1)...)
    for l1 in vec1.nzind
        r = l1 - 1
        j = 1
        Sl1 = ones(Int, N1)
        for i in 1 : N1 - 1
            k = searchsortedlast(F1[i][j], r)
            Sl1[D1[i][j][k]] = i + 1
            r -= F1[i][j][k]
            j = E1[i][j][k]
        end
        S1 = [Sl1[cumν1[i] + 1 : cumν1[i + 1]] for i in 1 : L1]
        start, = subdiagram(S1, ν, N1, E, D, B, F, T)
        for k in 1 : OM
            append!(I[k], @. φsum[k].nzind + start - 1)
            append!(V[k], vec1[l1] .* φsum[k].nzval)
        end
    end
    rtn = map(k -> sparsevec2(I[k], V[k], mult), 1 : OM)
    if perm
        f2 = N1 - f1
        f23 = f2 + N2 - f4
        p2 = Permutation([f2 + 1 : f23..., 1 : f2...])
        pp2 = CoxeterDecomposition(p2)
        for term in pp2.terms
            for k in 1 : OM
                rtn[k] = Papply2!(rtn[k], f1 + term, ν, N, E, D, F, T)
            end
        end
    end
    rtn
end

function _SDC(ν1, ν2, ν, E, D, B, F, T)
    S1 = map(α -> zeros(Int, α), collect(ν1))
    N1 = sum(ν1)
    N = sum(ν)
    j = 1
    k = 1
    for i in 1 : N1
        S1[j][k] = i
        if k == ν1[j]
            k = 1
            j += 1
        else
            k += 1
        end
    end

    columns = Vector{Int}[]
    k = N1 + 1
    for i in 1 : ν2[1]
        column = Int[]
        j = 1
        while j <= length(ν2) && i <= ν2[j]
            push!(column, k)
            j += 1
            k += 1
        end
        push!(columns, column)
    end

    reps = representatives(S1, columns, ν)
    start, dim, initj = subdiagram(S1, ν, N1, E, D, B, F, T)
    φs = antisymmetrize(reps, columns, ν, initj, dim, N1, N, E, D, F, T)

    len = length(φs)
    if len == 1
        Φ = φs
    else
        CS2 = zeros(len, len)
        for k in 1 : ν2[1] - 1
            φtemp = deepcopy(φs)
            for l in 1 : len
                φtemp[l] = Papply!(φtemp[l], columns[k][end], ν, initj, N1, N, E, D, F, T)
            end
            Ctemp = [dot(φs[k1], φtemp[k2]) for k1 in 1 : len, k2 in 1 : len] - (1.0 / length(columns[k])) * I
            mul!(CS2, Ctemp, Ctemp, 1.0, 1.0)
        end
        solutions = nullspace(CS2; atol = tol_nullspace)
        solutions = gaugefix!(solutions)
        Φ = [sum(φs[k] * solutions[k, l] for k in 1 : len) for l in 1 : size(solutions, 2)]
    end
    Φ, dim, initj
end

const TOL_GAUGE = 1e-11
# tolerance for gaugefixing should probably be bigger than that with which nullspace was determined

gaugefix!(C) = first(qrpos!(cref!(C, TOL_GAUGE)))
# gaugefix(C) = C*conj.(first(qrpos!(rref!(permutedims(C)))))

# Auxiliary tools
function qrpos!(C)
    q, r = qr(C)
    d = diag(r)
    map!(x-> x == zero(x) ? 1 : sign(x), d, d)
    D = Diagonal(d)
    Q = rmul!(Matrix(q), D)
    R = ldiv!(D, Matrix(r))
    return Q, R
end

function cref!(A::AbstractMatrix,
        ɛ = eltype(A) <: Union{Rational,Integer} ? 0 : 10*length(A)*eps(norm(A, Inf)))
    nr, nc = size(A)
    i = j = 1
    @inbounds while i <= nr && j <= nc
        (m, mj) = findabsmax(view(A, i, j:nc))
        mj = mj + j - 1
        if m <= ɛ
            if ɛ > 0
                A[i, j:nc] .= zero(eltype(A))
            end
            i += 1
        else
            @simd for k in i:nr
                A[k, j], A[k, mj] = A[k, mj], A[k, j]
            end
            d = A[i,j]
            @simd for k in i:nr
                A[k, j] /= d
            end
            for k in 1:nc
                if k != j
                    d = A[i, k]
                    @simd for l = i:nr
                        A[l, k] -= d*A[l, j]
                    end
                end
            end
            i += 1
            j += 1
        end
    end
    A
end

function findabsmax(a)
    isempty(a) && throw(ArgumentError("collection must be non-empty"))
    m = abs(first(a))
    mi = firstindex(a)
    for (k, v) in pairs(a)
        if abs(v) > m
            m = abs(v)
            mi = k
        end
    end
    return m, mi
end