"""
coeff = _3ν(ν1, ν2, ν)[τ′, τ]
computes the 3ν coefficient
"""
function _3ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν::SUNIrrep{Nc}) where Nc
    μ1, μ2, μ = map(collect ∘ weight, (ν1, ν2, ν))
    f1, f2 = map(sum, (μ1, μ2))
    μ .+= (f1 + f2 - sum(μ)) ÷ Nc
    _3ν(map(Tuple, (μ1, μ2, μ))...)
end

function _3ν(ν1, ν2, ν)
    vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
    vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
    right = SDC(ν1, ν2, ν, vec1, vec2; perm = true)
    left = SDC(ν2, ν1, ν, vec1, vec2)
    [dot(left[τ′], right[τ]) for τ′ in 1 : length(left), τ in 1 : length(right)]
end

"""
coeff = _6ν(ν1, ν2, ν12, ν3, ν, ν23)[τ23, τ′, τ12, τ]
computes the 6ν coefficient
"""
function _6ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν23::SUNIrrep{Nc}) where Nc
    μ1, μ2, μ12, μ3, μ, μ23 = map(collect ∘ weight, (ν1, ν2, ν12, ν3, ν, ν23))
    f1, f2, f3 = map(sum, (μ1, μ2, μ3))
    μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
    μ23 .+= (f2 + f3 - sum(μ23)) ÷ Nc
    μ .+= (sum(μ12) + f3 - sum(μ)) ÷ Nc
    _6ν(map(Tuple, (μ1, μ2, μ12, μ3, μ, μ23))...)
end

function _6ν(ν1, ν2, ν12, ν3, ν, ν23)
    vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
    vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
    vec3 = sparsevec2(Int128[1], [1.0], multiplicity(ν3))
    vec12 = SDC(ν1, ν2, ν12, vec1, vec2)
    right = [SDC(ν12, ν3, ν, vec12[τ12], vec3) for τ12 in 1 : length(vec12)]
    vec23 = SDC(ν2, ν3, ν23, vec2, vec3)
    left = [SDC(ν1, ν23, ν, vec1, vec23[τ23]) for τ23 in 1 : length(vec23)]
    [dot(left[τ23][τ′], right[τ12][τ]) for τ23 in 1 : length(vec23), τ′ in 1 : length(left[1]), τ12 in 1 : length(vec12), τ in 1 : length(right[1])]
end

function _6νrev(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν::SUNIrrep{Nc}, ν23::SUNIrrep{Nc}) where Nc
    A = _6ν(ν1, ν2, ν12, ν3, ν, ν23)
    a, b, c, d = size(A)
    Array(reshape(reshape(A, a * b, c * d)', c, d, a, b))
end

"""
coeff = _9ν(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)[τ13, τ24, τ′, τ12, τ34,　τ]
computes the 9ν coefficient
"""
function _9ν(ν1::SUNIrrep{Nc}, ν2::SUNIrrep{Nc}, ν12::SUNIrrep{Nc}, ν3::SUNIrrep{Nc}, ν4::SUNIrrep{Nc}, ν34::SUNIrrep{Nc}, ν13::SUNIrrep{Nc}, ν24::SUNIrrep{Nc}, ν::SUNIrrep{Nc}) where Nc
    μ1, μ2, μ12, μ3, μ4, μ34, μ13, μ24, μ = map(collect ∘ weight, (ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν))
    f1, f2, f3, f4 = map(sum, (μ1, μ2, μ3, μ4))
    μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
    μ34 .+= (f3 + f4 - sum(μ34)) ÷ Nc
    μ13 .+= (f1 + f3 - sum(μ13)) ÷ Nc
    μ24 .+= (f2 + f4 - sum(μ24)) ÷ Nc
    μ .+= (sum(μ12) + sum(μ34) - sum(μ)) ÷ Nc
    _9ν(map(Tuple, (μ1, μ2, μ12, μ3, μ4, μ34, μ13, μ24, μ))...)
end

function _9ν(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)
    N1 = isempty(ν1) ? 0 : sum(ν1)
    N4 = isempty(ν4) ? 0 : sum(ν4)
    vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
    vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
    vec3 = sparsevec2(Int128[1], [1.0], multiplicity(ν3))
    vec4 = sparsevec2(Int128[1], [1.0], multiplicity(ν4))
    vec12 = SDC(ν1, ν2, ν12, vec1, vec2)
    vec34 = SDC(ν3, ν4, ν34, vec3, vec4)
    right = [SDC(ν12, ν34, ν, vec12[τ12], vec34[τ34]; perm = true, f1 = N1, f4 = N4) for τ12 in 1 : length(vec12), τ34 in 1 : length(vec34)]
    vec13 = SDC(ν1, ν3, ν13, vec1, vec3)
    vec24 = SDC(ν2, ν4, ν24, vec2, vec4)
    left = [SDC(ν13, ν24, ν, vec13[τ13], vec24[τ24]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24)]
    [dot(left[τ13, τ24][τ′], right[τ12, τ34][τ]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24), τ′ in 1 : length(left[1, 1]), τ12 in 1 : length(vec12), τ34 in 1 : length(vec34), τ in 1 : length(right[1, 1])]
end

"""
tables = table_9ν(Nc, widthmax, table4, table_3ν)
creates a table of 9ν symbols
"""
function table_9ν(Nc, widthmax, table4, table_3ν)
    adjoint = adjointirrep(Val(Nc))
    funda = fundamentalirrep(Val(Nc))
    trivial = trivialirrep(Val(Nc))

    λ = irreplist(Nc, widthmax)
    table1 = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Matrix{Float64}}()
    μ2, μ3, μ4, μ24, μ34 = map(collect ∘ weight, (funda, adjoint, trivial, funda, adjoint))
    f2, f3, f4 = map(sum, (μ2, μ3, μ4))
    μ24 .+= (f2 + f4 - sum(μ24)) ÷ Nc
    μ34 .+= (f3 + f4 - sum(μ34)) ÷ Nc
    ν2, ν3, ν4, ν24, ν34 = map(Tuple, (μ2, μ3, μ4, μ24, μ34))
    N4 = isempty(ν4) ? 0 : sum(ν4)
    vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
    vec3 = sparsevec2(Int128[1], [1.0], multiplicity(ν3))
    vec4 = sparsevec2(Int128[1], [1.0], multiplicity(ν4))
    vec24 = SDC(ν2, ν4, ν24, vec2, vec4)
    vec34 = SDC(ν3, ν4, ν34, vec3, vec4)
    for α1 in λ
        β1dict = directproduct(α1, funda)
        for β1 in keys(β1dict)
            if weight(β1)[1] <= widthmax
                μ1, μ12 = map(collect ∘ weight, (α1, β1))
                f1 = sum(μ1)
                μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
                ν1, ν12 = map(Tuple, (μ1, μ12))
                N1 = isempty(ν1) ? 0 : sum(ν1)
                vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
                vec12 = SDC(ν1, ν2, ν12, vec1, vec2)

                α2dict = directproduct(α1, adjoint)
                for α2 in keys(α2dict)
                    if weight(α2)[1] <= widthmax
                        μ13 = (collect ∘ weight)(α2)
                        μ13 .+= (f1 + f3 - sum(μ13)) ÷ Nc
                        ν13 = Tuple(μ13)
                        vec13 = SDC(ν1, ν3, ν13, vec1, vec3)

                        β2dict1 = directproduct(β1, adjoint)
                        β2dict2 = directproduct(α2, funda)
                        for β2 in keys(β2dict1) ∩ keys(β2dict2)
                            if weight(β2)[1] <= widthmax
                                μ = (collect ∘ weight)(β2)
                                μ .+= (sum(μ12) + sum(μ34) - sum(μ)) ÷ Nc
                                ν = Tuple(μ)
                                right = [SDC(ν12, ν34, ν, vec12[τ12], vec34[τ34]; perm = true, f1 = N1, f4 = N4) for τ12 in 1 : length(vec12), τ34 in 1 : length(vec34)]
                                left = [SDC(ν13, ν24, ν, vec13[τ13], vec24[τ24]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24)]
                                symbol = [dot(left[τ13, τ24][τ′], right[τ12, τ34][τ]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24), τ′ in 1 : length(left[1, 1]), τ12 in 1 : length(vec12), τ34 in 1 : length(vec34), τ in 1 : length(right[1, 1])]
                                table1[α1, β1, α2, β2] = symbol[:, 1, 1, 1, 1, :]
                            end
                        end
                    end
                end
            end
        end
    end

    table2 = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Vector{Float64}}()
    μ2, μ3, μ4, μ24, μ34 = map(collect ∘ weight, (funda, trivial, adjoint, funda, adjoint))
    f2, f3, f4 = map(sum, (μ2, μ3, μ4))
    μ24 .+= (f2 + f4 - sum(μ24)) ÷ Nc
    μ34 .+= (f3 + f4 - sum(μ34)) ÷ Nc
    ν2, ν3, ν4, ν24, ν34 = map(Tuple, (μ2, μ3, μ4, μ24, μ34))
    N4 = isempty(ν4) ? 0 : sum(ν4)
    vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
    vec3 = sparsevec2(Int128[1], [1.0], multiplicity(ν3))
    vec4 = sparsevec2(Int128[1], [1.0], multiplicity(ν4))
    vec24 = SDC(ν2, ν4, ν24, vec2, vec4)
    vec34 = SDC(ν3, ν4, ν34, vec3, vec4)
    for α in λ
        μ1, μ13 = map(collect ∘ weight, (α, α))
        f1 = sum(μ1)
        μ13 .+= (f1 + f3 - sum(μ13)) ÷ Nc
        ν1, ν13 = map(Tuple, (μ1, μ13))
        N1 = isempty(ν1) ? 0 : sum(ν1)
        vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
        vec13 = SDC(ν1, ν3, ν13, vec1, vec3)

        βdict = directproduct(α, funda)
        for β1 in keys(βdict)
            if weight(β1)[1] <= widthmax
                μ12 = (collect ∘ weight)(β1)
                μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
                ν12 = Tuple(μ12)
                vec12 = SDC(ν1, ν2, ν12, vec1, vec2)

                β2dict = directproduct(β1, adjoint)
                for β2 in keys(β2dict) ∩ keys(βdict)
                    if weight(β2)[1] <= widthmax
                        μ = (collect ∘ weight)(β2)
                        μ .+= (sum(μ12) + sum(μ34) - sum(μ)) ÷ Nc
                        ν = Tuple(μ)
                        right = [SDC(ν12, ν34, ν, vec12[τ12], vec34[τ34]; perm = true, f1 = N1, f4 = N4) for τ12 in 1 : length(vec12), τ34 in 1 : length(vec34)]
                        left = [SDC(ν13, ν24, ν, vec13[τ13], vec24[τ24]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24)]
                        symbol = [dot(left[τ13, τ24][τ′], right[τ12, τ34][τ]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24), τ′ in 1 : length(left[1, 1]), τ12 in 1 : length(vec12), τ34 in 1 : length(vec34), τ in 1 : length(right[1, 1])]
                        table2[α, β1, β2] = symbol[1, 1, 1, 1, 1, :]
                    end
                end
            end
        end
    end

    table3 = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Vector{Float64}}()
    μ2, μ3, μ4, μ24, μ34 = map(collect ∘ weight, (funda, adjoint, adjoint, funda, trivial))
    f2, f3, f4 = map(sum, (μ2, μ3, μ4))
    μ24 .+= (f2 + f4 - sum(μ24)) ÷ Nc
    μ34 .+= (f3 + f4 - sum(μ34)) ÷ Nc
    ν2, ν3, ν4, ν24, ν34 = map(Tuple, (μ2, μ3, μ4, μ24, μ34))
    N4 = isempty(ν4) ? 0 : sum(ν4)
    vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
    vec3 = sparsevec2(Int128[1], [1.0], multiplicity(ν3))
    vec4 = sparsevec2(Int128[1], [1.0], multiplicity(ν4))
    vec24 = SDC(ν2, ν4, ν24, vec2, vec4)
    vec34 = SDC(ν3, ν4, ν34, vec3, vec4)
    for α1 in λ
        α2dict = directproduct(α1, adjoint)
        for α2 in keys(α2dict)
            if weight(α2)[1] <= widthmax
                μ1, μ13 = map(collect ∘ weight, (α1, α2))
                f1 = sum(μ1)
                μ13 .+= (f1 + f3 - sum(μ13)) ÷ Nc
                ν1, ν13 = map(Tuple, (μ1, μ13))
                N1 = isempty(ν1) ? 0 : sum(ν1)
                vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
                vec13 = SDC(ν1, ν3, ν13, vec1, vec3)

                β1dict = directproduct(α1, funda)
                β2dict = directproduct(α2, funda)
                for β in keys(β1dict) ∩ keys(β2dict)
                    if weight(β)[1] <= widthmax
                        μ12 = (collect ∘ weight)(β)
                        μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
                        ν12 = Tuple(μ12)
                        vec12 = SDC(ν1, ν2, ν12, vec1, vec2)
                        μ = (collect ∘ weight)(β)
                        μ .+= (sum(μ12) + sum(μ34) - sum(μ)) ÷ Nc
                        ν = Tuple(μ)
                        right = [SDC(ν12, ν34, ν, vec12[τ12], vec34[τ34]; perm = true, f1 = N1, f4 = N4) for τ12 in 1 : length(vec12), τ34 in 1 : length(vec34)]
                        left = [SDC(ν13, ν24, ν, vec13[τ13], vec24[τ24]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24)]
                        symbol = [dot(left[τ13, τ24][τ′], right[τ12, τ34][τ]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24), τ′ in 1 : length(left[1, 1]), τ12 in 1 : length(vec12), τ34 in 1 : length(vec34), τ in 1 : length(right[1, 1])]
                        table3[α1, β, α2] = symbol[:, 1, 1, 1, 1, 1]
                    end
                end
            end
        end
    end

    for h in 0 : Nc - 1
        if iseven(Nc) && isodd(h)
            continue
        end
        γ = SUNIrrep(ntuple(i -> 0 + (i <= h), Val(Nc)))
        T = OM_matrix(λ, γ) .> 0
        for (i, α1) in Iterators.filter(x -> any(T[x[1], :]), enumerate(λ))
            for (j, β1) in Iterators.filter(x -> T[i, x[1]], enumerate(λ))
                comp = sum(weight(α1)) - sum(weight(β1))
                if (h % Nc == 0 || comp % Nc == 0) && !(comp > 0 || (comp == 0 && i >= j))
                    table_3ν[α1, β1, γ] = inv(table_3ν[β1, α1, γ])
                end
            end
        end
    end

    factor1 = _3ν(adjoint, adjoint, trivial)[1, 1]
    for h in 0 : Nc - 1
        if iseven(Nc) && isodd(h)
            continue
        end
        γ = SUNIrrep(ntuple(i -> 0 + (i <= h), Val(Nc)))
        T = OM_matrix(λ, γ) .> 0
        for (i, α1) in Iterators.filter(x -> any(T[x[1], :]), enumerate(λ))
            α2dict = directproduct(α1, adjoint)
            for (j, β1) in Iterators.filter(x -> T[i, x[1]], enumerate(λ))
                β2dict = directproduct(β1, adjoint)

                comp = sum(weight(α1)) - sum(weight(β1))
                if (h % Nc == 0 || comp % Nc == 0) && !(comp > 0 || (comp == 0 && i >= j))
                    factor2 = table_3ν[α1, β1, γ]
                    for α2 in keys(α2dict)
                        if weight(α2)[1] <= widthmax
                            for β2 in keys(β2dict)
                                if weight(β2)[1] <= widthmax
                                    k = findfirst(isequal(α2), λ)
                                    l = findfirst(isequal(β2), λ)
                                    if T[k, l]
                                        factor3 = inv(table_3ν[α2, β2, γ])
                                        A = table4[β1, α1, γ, β2, α2]
                                        @tensor B[k2, k3, k4, k1] := factor3[k4, k6] * A[k3, k2, k6, k5] * factor2[k5, k1]
                                        table4[α1, β1, γ, α2, β2] = factor1 .* B
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    table_6ν = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Matrix{Float64}}()
    let γ = trivial
        T = OM_matrix(λ, γ) .> 0
        for (i, αi) in enumerate(λ), (l, βl) in enumerate(λ)
            if T[i, l]
                βkdict = directproduct(funda, αi)
                for βk in keys(βkdict)
                    if weight(βk)[1] <= widthmax
                        k = findfirst(isequal(βk), λ)
                        for j in findall(T[k, :])
                            αj = λ[j]
                            if outer_multiplicity(αj, funda, βl) > 0
                                comp = sum(weight(αj)) - sum(weight(αi))
                                if comp > 0 || (comp == 0 && j >= i)
                                    table_6ν[αj, βl, αi, γ, βk] = _6ν(αj, funda, βl, αi, γ, βk)[1, :, 1, :]
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    let γ = trivial
        T = OM_matrix(λ, γ) .> 0
        for (i, αi) in enumerate(λ), (l, βl) in enumerate(λ)
            if T[i, l]
                βkdict = directproduct(funda, αi)
                for βk in keys(βkdict)
                    if weight(βk)[1] <= widthmax
                        k = findfirst(isequal(βk), λ)
                        for j in findall(T[k, :])
                            αj = λ[j]
                            if outer_multiplicity(αj, funda, βl) > 0
                                comp = sum(weight(αj)) - sum(weight(αi))
                                if !(comp > 0 || (comp == 0 && j >= i))
                                    table_6ν[αj, βl, αi, γ, βk] = inv(table_3ν[αj, βk, γ]) * table_6ν[αi, βk, αj, γ, βl]' * table_3ν[βl, αi, γ] .* (_3ν(αj, funda, βl)[1, 1] / _3ν(αi, funda, βk)[1, 1])
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    table5 = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Matrix{Float64}}()
    let γ = trivial
        T = OM_matrix(λ, γ) .> 0
        for (i, αi) in enumerate(λ), (l, βl) in enumerate(λ)
            if T[i, l]
                βkdict = directproduct(funda, αi)
                for βk in keys(βkdict)
                    if weight(βk)[1] <= widthmax
                        factor2 = _3ν(αi, funda, βk)[1, 1]
                        k = findfirst(isequal(βk), λ)
                        for j in findall(T[k, :])
                            αj = λ[j]
                            if outer_multiplicity(αj, funda, βl) > 0
                                factor1 = table_6ν[αj, βl, αi, γ, βk]
                                table5[αj, βl, αi, γ, βk] = factor1 .* factor2
                            end
                        end
                    end
                end
            end
        end
    end
    table6 = table_3ν

    table1, table2, table3, table4, table5, table6
end