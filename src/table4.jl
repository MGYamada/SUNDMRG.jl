"""
make_table4(Nc, widthmax)
Making tables[4] dictionary. This function is supposed to be run on the supercomputer.
"""
function make_table4(Nc, widthmax)
    MPI.Init()

    comm = MPI.COMM_WORLD
    size = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    adjoint = adjointirrep(Val(Nc))
    trivial = trivialirrep(Val(Nc))

    λ = irreplist(Nc, widthmax)
    if rank == 0
        table4 = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Array{Float64, 4}}()
        o1s = [Int[] for i in 1 : size]
        o2s = [Int[] for i in 1 : size]
        o3s = [Int[] for i in 1 : size]
        o4s = [Int[] for i in 1 : size]
        names = [Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}[] for i in 1 : size]
    end
    c = 0
    αβγ = Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}[]

    for h in 0 : Nc - 1
        if iseven(Nc) && isodd(h)
            continue
        end
        γ = SUNIrrep(ntuple(i -> 0 + (i <= h), Val(Nc)))
        OM = OM_matrix(λ, γ)
        for (i, α1) in Iterators.filter(x -> any(OM[x[1], :] .> 0), enumerate(λ))
            α2dict = directproduct(α1, adjoint)
            for (j, β1) in Iterators.filter(x -> OM[i, x[1]] > 0, enumerate(λ))
                β2dict = directproduct(β1, adjoint)

                comp = sum(weight(α1)) - sum(weight(β1))
                if (h % Nc == 0 || comp % Nc == 0) && (comp > 0 || (comp == 0 && i >= j))
                    if rank == 0
                        α2s = sort!(filter!(x -> weight(x)[1] <= widthmax, collect(keys(α2dict)))) # for safety
                        β2s = sort!(filter!(x -> weight(x)[1] <= widthmax, collect(keys(β2dict)))) # for safety
                        for α2 in α2s
                            for β2 in β2s
                                k = findfirst(isequal(α2), λ)
                                l = findfirst(isequal(β2), λ)
                                if OM[k, l] > 0
                                    push!(o1s[c % size + 1], α2dict[α2])
                                    push!(o2s[c % size + 1], β2dict[β2])
                                    push!(o3s[c % size + 1], OM[k, l])
                                    push!(o4s[c % size + 1], OM[i, j])
                                    push!(names[c % size + 1], (α1, β1, γ, α2, β2))
                                end
                            end
                        end
                    end
                    if c % size == rank
                        push!(αβγ, (α1, β1, γ))
                    end
                    c += 1
                end
            end
        end
    end

    rtn = Array{Float64, 4}[]

    μ3, μ4, μ34 = map(collect ∘ weight, (adjoint, adjoint, trivial))
    f3, f4 = map(sum, (μ3, μ4))
    μ34 .+= (f3 + f4 - sum(μ34)) ÷ Nc
    ν3, ν4, ν34 = map(Tuple, (μ3, μ4, μ34))
    N4 = isempty(ν4) ? 0 : sum(ν4)
    vec3 = sparsevec2(Int128[1], [1.0], multiplicity(ν3))
    vec4 = sparsevec2(Int128[1], [1.0], multiplicity(ν4))
    vec34 = SDC(ν3, ν4, ν34, vec3, vec4)
    for (α1, β1, γ) in αβγ
        α2dict = directproduct(α1, adjoint)
        β2dict = directproduct(β1, adjoint)
        α2s = sort!(filter!(x -> weight(x)[1] <= widthmax, collect(keys(α2dict)))) # for safety
        β2s = sort!(filter!(x -> weight(x)[1] <= widthmax, collect(keys(β2dict)))) # for safety

        μ1, μ2, μ12 = map(collect ∘ weight, (α1, β1, γ))
        f1, f2 = map(sum, (μ1, μ2))
        μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
        ν1, ν2, ν12 = map(Tuple, (μ1, μ2, μ12))
        N1 = isempty(ν1) ? 0 : sum(ν1)
        vec1 = sparsevec2(Int128[1], [1.0], multiplicity(ν1))
        vec2 = sparsevec2(Int128[1], [1.0], multiplicity(ν2))
        vec12 = SDC(ν1, ν2, ν12, vec1, vec2)
        μ = (collect ∘ weight)(γ)
        μ .+= (sum(μ12) + sum(μ34) - sum(μ)) ÷ Nc
        ν = Tuple(μ)
        right = [SDC(ν12, ν34, ν, vec12[τ12], vec34[τ34]; perm = true, f1 = N1, f4 = N4) for τ12 in 1 : length(vec12), τ34 in 1 : length(vec34)]

        ν24 = Dict{SUNIrrep{Nc}, Tuple}()
        vec24 = Dict{SUNIrrep{Nc}, Vector{SparseVector2{Float64, Int128}}}()
        for β2 in β2s
            μ24 = (collect ∘ weight)(β2)
            μ24 .+= (f2 + f4 - sum(μ24)) ÷ Nc
            ν24[β2] = Tuple(μ24)
            vec24[β2] = SDC(ν2, ν4, ν24[β2], vec2, vec4)
        end

        for α2 in α2s
            μ13 = (collect ∘ weight)(α2)
            μ13 .+= (f1 + f3 - sum(μ13)) ÷ Nc
            ν13 = Tuple(μ13)
            vec13 = SDC(ν1, ν3, ν13, vec1, vec3)

            for β2 in β2s
                if outer_multiplicity(α2, β2, γ) > 0
                    left = [SDC(ν13, ν24[β2], ν, vec13[τ13], vec24[β2][τ24]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24[β2])]
                    symbol = [dot(left[τ13, τ24][τ′], right[τ12, τ34][τ]) for τ13 in 1 : length(vec13), τ24 in 1 : length(vec24[β2]), τ′ in 1 : length(left[1, 1]), τ12 in 1 : length(vec12), τ34 in 1 : length(vec34), τ in 1 : length(right[1, 1])]
                    push!(rtn, symbol[:, :, :, :, 1, 1])
                end
            end
        end
    end
    println("rank $rank: finished!")
    if isempty(rtn)
        sendbuf = Float64[]
    else
        sendbuf = vcat(vec.(rtn)...)
    end
    if rank == 0
        counts = [sum(@. m * n * l * o) for (m, n, l, o) in zip(o1s, o2s, o3s, o4s)]
        recvbuf = zeros(sum(counts))
        MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), 0, comm)
    else
        MPI.Gatherv!(sendbuf, nothing, 0, comm)
    end

    if rank == 0
        cumcounts = vcat(0, cumsum(counts)...)
        for i in 1 : length(names)
            cum2 = vcat(0, cumsum(@. o1s[i] * o2s[i] * o3s[i] * o4s[i])...)
            for j in 1 : length(names[i])
                table4[names[i][j]] = reshape(recvbuf[cumcounts[i] + 1 : cumcounts[i + 1]][cum2[j] + 1 : cum2[j + 1]], o1s[i][j], o2s[i][j], o3s[i][j], o4s[i][j])
            end
        end

        @save "table4half_SU$(Nc)_$widthmax.jld2" table4
        println("All finished!")
    end

    MPI.Finalize()
end