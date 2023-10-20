"""
multiplicity(ν)::Int128
computes the multiplicity or the number of SYTx.
"""
function multiplicity(ν)::Int128
    if isempty(ν)
        return one(Int128)
    end
    hook = Int[]
    for i in 1 : length(ν)
        for j in 1 : ν[i]
            push!(hook, ν[i] - j + count(ν[i : end] .>= j))
        end
    end
    Int128(factorial(big(sum(ν))) ÷ prod(big.(hook)))
end

"""
V, E, D = SYTdiagram(γ)
generates a reverse Wilf-Rao-Shanker diagram
"""
function SYTdiagram(γ)
    vertex = [[collect(γ)]]
    edge = Vector{Vector{Int}}[]
    drop = Vector{Vector{Int}}[]
    f12 = sum(γ)
    for i in 1 : f12 - 1
        push!(vertex, Vector{Vector{Int}}[])
        push!(edge, Vector{Int}[])
        push!(drop, Vector{Int}[])
        for j in 1 : length(vertex[i])
            push!(edge[i], Int[])
            push!(drop[i], Int[])
            for k in 1 : length(vertex[i][j])
                subγ = copy(vertex[i][j])
                position = k == 1 ? subγ[k] : sum(γ[1 : k - 1]) + subγ[k]
                if subγ[k] > 1
                    if k == length(vertex[i][j]) || subγ[k] > subγ[k + 1]
                        subγ[k] -= 1
                        test = findfirst(isequal(subγ), vertex[i + 1])
                        if isnothing(test)
                            push!(vertex[i + 1], subγ)
                            push!(edge[i][j], length(vertex[i + 1]))
                        else
                            push!(edge[i][j], test)
                        end
                        push!(drop[i][j], position)
                    end
                else
                    if k == length(vertex[i][j])
                        popat!(subγ, k)
                        test = findfirst(isequal(subγ), vertex[i + 1])
                        if isnothing(test)
                            push!(vertex[i + 1], subγ)
                            push!(edge[i][j], length(vertex[i + 1]))
                        else
                            push!(edge[i][j], test)
                        end
                        push!(drop[i][j], position)
                    end
                end
            end
        end
    end
    invvertex = reverse(vertex)
    invedge = Vector{Vector{Int}}[]
    invdrop = Vector{Vector{Int}}[]
    for i in 1 : f12 - 1
        push!(invedge, Vector{Int}[])
        push!(invdrop, Vector{Int}[])
        for j in 1 : length(invvertex[i])
            push!(invedge[i], Int[])
            push!(invdrop[i], Int[])
        end
        for j in 1 : length(edge[f12 - i])
            for k in 1 : length(edge[f12 - i][j])
                push!(invedge[i][edge[f12 - i][j][k]], j)
                push!(invdrop[i][edge[f12 - i][j][k]], drop[f12 - i][j][k])
            end
        end
    end
    invvertex, invedge, invdrop
end

"""
B, F = bf(N, V, E, T)
generates a dictionary.
"""
function bf(N, V, E, T)
    b = [zeros(T, length(V[i])) for i in 1 : N]
    b[N][1] = 1
    for i in N - 1 : -1 : 1
        for j in 1 : length(V[i])
            b[i][j] = sum(b[i + 1][E[i][j]])
        end
    end
    f = [[zeros(T, length(E[i][j])) for j in 1 : length(E[i])] for i in 1 : length(E)]
    for i in 1 : N - 1
        for j in 1 : length(E[i])
            f[i][j][1] = zero(T)
            bsum = zero(T)
            for k in 2 : length(E[i][j])
                bsum += b[i + 1][E[i][j][k - 1]]
                f[i][j][k] = bsum
            end
        end
    end
    b, f
end

"""
start, dim, initj = subdiagram(S1, γ, N1, E, D, B, F, T)
returns the subspace for the subdiagram (start : start + dim - 1)
"""
function subdiagram(S1, γ, N1, E, D, B, F, T)
    S1embedded = map(α -> zeros(Int, α), collect(γ))
    for i in 1 : length(S1)
        for j in 1 : length(S1[i])
            S1embedded[i][j] = S1[i][j]
        end
    end
    S1list = vcat(S1embedded...)
    j = 1
    startind = zero(T)
    for i in 1 : N1 - 1
        position = findfirst(isequal(i + 1), S1list)
        k = findfirst(isequal(position), D[i][j])
        startind += F[i][j][k]
        j = E[i][j][k]
    end
    startind + 1, B[N1][j], j
end

"""
vec = P!(factor, axial, vec, l, γ, initj, N1, N, E, D, F, T)
applys P_{l, l + 1} and transforms the basis
"""
function P!(factor, axial, vec, l, γ, initj, N1, N, E, D, F, T)
    len = length(vec.nzind)
    I = Vector{T}(undef, len)
    V = Vector{Float64}(undef, len)
    Cplus = factor * (1.0 + axial)
    Cminus = factor * (1.0 - axial)
    factor2 = factor * axial
    @inbounds Threads.@threads for ind in 1 : len
        u1 = vec.nzind[ind]
        r = u1 - 1
        positions = ones(Int, N)
        j = initj
        for i in N1 : N - 1
            k = searchsortedlast(F[i][j], r)
            positions[i + 1] = D[i][j][k]
            r -= F[i][j][k]
            j = E[i][j][k]
        end
        m = positions[l]
        m1 = 0
        m2 = 0
        temp = m
        while temp > 0
            m1 += 1
            m2 = temp
            temp -= γ[m1]
        end
        n = positions[l + 1]
        n1 = 0
        n2 = 0
        temp = n
        while temp > 0
            n1 += 1
            n2 = temp
            temp -= γ[n1]
        end
        if m1 == n1
            I[ind] = 0
            vec.nzval[ind] *= Cplus
        elseif m2 == n2
            I[ind] = 0
            vec.nzval[ind] *= Cminus
        else
            positions[l], positions[l + 1] = positions[l + 1], positions[l]
            j = initj
            t = zero(T)
            for i in N1 : N - 1
                k = findfirst(isequal(positions[i + 1]), D[i][j])
                t += F[i][j][k]
                j = E[i][j][k]
            end
            ρ = 1.0 / ((m1 - n1) - (m2 - n2)) # be careful
            I[ind] = t + 1
            V[ind] = factor2 * sqrt(1.0 - ρ ^ 2) * vec.nzval[ind]
            vec.nzval[ind] *= factor * (1.0 + ρ * axial)
        end
    end
    flag = I .> 0
    vec + sparsevec2(I[flag], V[flag], vec.n)
end

"""
vec = Papply!(vec, l, γ, initj, N1, N, E, D, F, T)
applys P_{l, l + 1}
"""
function Papply!(vec, l, γ, initj, N1, N, E, D, F, T)
    len = length(vec.nzind)
    I = Vector{T}(undef, len)
    V = Vector{Float64}(undef, len)
    @inbounds Threads.@threads for ind in 1 : len
        u1 = vec.nzind[ind]
        r = u1 - 1
        positions = ones(Int, N)
        j = initj
        for i in N1 : N - 1
            k = searchsortedlast(F[i][j], r)
            positions[i + 1] = D[i][j][k]
            r -= F[i][j][k]
            j = E[i][j][k]
        end
        m = positions[l]
        m1 = 0
        m2 = 0
        temp = m
        while temp > 0
            m1 += 1
            m2 = temp
            temp -= γ[m1]
        end
        n = positions[l + 1]
        n1 = 0
        n2 = 0
        temp = n
        while temp > 0
            n1 += 1
            n2 = temp
            temp -= γ[n1]
        end
        if m1 == n1
            I[ind] = 0
        elseif m2 == n2
            I[ind] = 0
            vec.nzval[ind] *= -1.0
        else
            positions[l], positions[l + 1] = positions[l + 1], positions[l]
            j = initj
            t = zero(T)
            for i in N1 : N - 1
                k = findfirst(isequal(positions[i + 1]), D[i][j])
                t += F[i][j][k]
                j = E[i][j][k]
            end
            ρ = 1.0 / ((m1 - n1) - (m2 - n2)) # be careful
            I[ind] = t + 1
            V[ind] = sqrt(1.0 - ρ ^ 2) * vec.nzval[ind]
            vec.nzval[ind] *= ρ
        end
    end
    flag = I .> 0
    vec + sparsevec2(I[flag], V[flag], vec.n)
end

"""
vec = Papply2!(vec, l, γ, N, E, D, F, T)
applys P_{l, l + 1}
"""
function Papply2!(vec, l, γ, N, E, D, F, T)
    len = length(vec.nzind)
    I = Vector{T}(undef, len)
    V = Vector{Float64}(undef, len)
    @inbounds Threads.@threads for ind in 1 : len
        u1 = vec.nzind[ind]
        r = u1 - 1
        positions = ones(Int, N)
        j = 1
        for i in 1 : N - 1
            k = searchsortedlast(F[i][j], r)
            positions[i + 1] = D[i][j][k]
            r -= F[i][j][k]
            j = E[i][j][k]
        end
        m = positions[l]
        m1 = 0
        m2 = 0
        temp = m
        while temp > 0
            m1 += 1
            m2 = temp
            temp -= γ[m1]
        end
        n = positions[l + 1]
        n1 = 0
        n2 = 0
        temp = n
        while temp > 0
            n1 += 1
            n2 = temp
            temp -= γ[n1]
        end
        if m1 == n1
            I[ind] = 0
        elseif m2 == n2
            I[ind] = 0
            vec.nzval[ind] *= -1.0
        else
            positions[l], positions[l + 1] = positions[l + 1], positions[l]
            j = 1
            t = zero(T)
            for i in 1 : N - 1
                k = findfirst(isequal(positions[i + 1]), D[i][j])
                t += F[i][j][k]
                j = E[i][j][k]
            end
            ρ = 1.0 / ((m1 - n1) - (m2 - n2)) # be careful
            I[ind] = t + 1
            V[ind] = sqrt(1.0 - ρ ^ 2) * vec.nzval[ind]
            vec.nzval[ind] *= ρ
        end
    end
    flag = I .> 0
    vec + sparsevec2(I[flag], V[flag], vec.n)
end