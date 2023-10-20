"""
make_table3nu(Nc, widthmax)
Making table3nu dictionary. This function is supposed to be run on the supercomputer.
"""
function make_table3nu(Nc, widthmax)
    MPI.Init()

    comm = MPI.COMM_WORLD
    Ncpu = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    λ = irreplist(Nc, widthmax)
    # flag1 = falses(length(λ))
    # abcdict = Dict{SUNIrrep{Nc}, Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}}()
    # i = 0
    # while !all(flag1)
    #     abc = irreplist(Nc, i)
    #     for a in abc, b in abc, c in abc
    #         for ab in keys(directproduct(a, b))
    #             αdict = directproduct(ab, c)
    #             for α in keys(αdict)
    #                 j = findfirst(isequal(α), λ)
    #                 if !isnothing(j) && (!flag1[j] || maximum(sum.(weight.((a, b, c)))) < maximum(sum.(weight.(abcdict[α][1 : 3]))))
    #                     abcdict[α] = (a, b, c, ab)
    #                     flag1[j] = true
    #                 end
    #             end
    #         end
    #     end
    #     i += 1
    # end

    # flag2 = falses(length(λ))
    # dedict = Dict{SUNIrrep{Nc}, Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}}}()
    # i = 0
    # while !all(flag2)
    #     de = irreplist(Nc, i)
    #     for d in de, e in de
    #         βdict = directproduct(d, e)
    #         for β in keys(βdict)
    #             j = findfirst(isequal(β), λ)
    #             if !isnothing(j) && (!flag2[j] || maximum(sum.(weight.((d, e)))) < maximum(sum.(weight.(dedict[β]))))
    #                 dedict[β] = (d, e)
    #                 flag2[j] = true
    #             end
    #         end
    #     end
    #     i += 1
    # end

    if rank == 0
        table_3ν = Dict{Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}, Matrix{Float64}}()
        names = [Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}[] for i in 1 : Ncpu]
        OMs = [Int[] for i in 1 : Ncpu]
    end
    c = 0
    αβγ = Tuple{SUNIrrep{Nc}, SUNIrrep{Nc}, SUNIrrep{Nc}}[]

    for h in 0 : Nc
        if iseven(Nc) && isodd(h)
            continue
        end
        γ = SUNIrrep(ntuple(i -> 0 + (i <= h), Val(Nc)))
        for (i, α1) in enumerate(λ), (j, β1) in enumerate(λ)
            OM = outer_multiplicity(α1, β1, γ)
            if OM > 0
                comp = sum(weight(α1)) - sum(weight(β1))
                if (h % Nc == 0 || comp % Nc == 0) && (comp > 0 || (comp == 0 && i >= j))
                    if rank == 0
                        push!(names[c % Ncpu + 1], (α1, β1, γ))
                        push!(OMs[c % Ncpu + 1], OM)
                    end
                    if c % Ncpu == rank
                        push!(αβγ, (α1, β1, γ))
                    end
                    c += 1
                end
            end
        end
    end

    rtn = Matrix{Float64}[]

    for (α1, β1, γ) in αβγ
        # a, b, c, ab = abcdict[α1]
        # d, e = dedict[β1]
        # β1dict = directproduct(d, e)
        # flag4 = false
        # for (ρ, σ) in sort!(vec(collect(Iterators.product(keys(directproduct(ab, d)), keys(directproduct(c, e))))); by = x -> maximum(sum.(weight.(x))))
        #     if haskey(directproduct(ρ, σ), γ)
        #         dadict = directproduct(d, a)
        #         flag3 = false
        #         for da in keys(dadict)
        #             if haskey(directproduct(da, b), ρ)
        #                 D1 = _9ν(ab, c, α1, d, e, β1, ρ, σ, γ)
        #                 D2 = _3ν(ab, d, ρ)
        #                 @tensor D3[k7, k5, k6, k1, k2, k3] := D2[k7, k4] * D1[k4, k5, k6, k1, k2, k3]
        #                 D4 = _3ν(c, e, σ)
        #                 @tensor D5[k7, k8, k6, k1, k2, k3] := D4[k8 , k5] * D3[k7, k5, k6, k1, k2, k3]
        #                 D6 = _6νrev(d, a, da, b, ρ, ab)
        #                 @tensor D[k9, k10, k8, k6, k0, k1, k2, k3] := D6[k9, k10, k0, k7] * D5[k7, k8, k6, k1, k2, k3]

        #                 C = zeros(size(D)...)
        #                 OM = size(C, 8)
        #                 νdict = directproduct(b, c)
        #                 for ν in keys(νdict)
        #                     α1dict = directproduct(a, ν)
        #                     if haskey(α1dict, α1)
        #                         C1 = _6ν(a, b, ab, c, α1, ν)
        #                         B = zeros(size(D)[1 : 4]..., α1dict[α1], νdict[ν], OM, β1dict[β1])
        #                         μdict1 = directproduct(da, e)
        #                         μdict2 = directproduct(β1, a)
        #                         for μ in keys(μdict1) ∩ keys(μdict2)
        #                             if haskey(directproduct(μ, ν), γ)
        #                                 B1 = _6νrev(β1, a, μ, ν, γ, α1)
        #                                 A = zeros(dadict[da], μdict1[μ], β1dict[β1], μdict2[μ])
        #                                 for ea in keys(directproduct(e, a))
        #                                     if haskey(directproduct(d, ea), μ)
        #                                         A1 = _6ν(d, e, β1, a, μ, ea)
        #                                         A2 = _3ν(e, a, ea)
        #                                         @tensor A3[k10, k9, k2, k6] := A2[k10, k8] * A1[k8, k9, k2, k6]
        #                                         A4 = _6νrev(d, a, da, e, μ, ea)
        #                                         @tensor A[k11, k12, k2, k6] += A4[k11, k12, k10, k9] * A3[k10, k9, k2, k6]
        #                                     end
        #                                 end
        #                                 @tensor B2[k11, k12, k7, k5, k3, k2] := A[k11, k12, k2, k6] * B1[k6, k7, k5, k3]
        #                                 B3 = _9ν(da, e, μ, b, c, ν, ρ, σ, γ)
        #                                 @tensor B[k11, k13, k14, k15, k5, k4, k3, k2] += B3[k13, k14, k15, k12, k4, k7] * B2[k11, k12, k7, k5, k3, k2]
        #                             end
        #                         end
        #                         @tensor C[k11, k13, k14, k15, k0, k1, k2, k3] += B[k11, k13, k14, k15, k5, k4, k3, k2] * C1[k4, k5, k0, k1]
        #                     end
        #                 end
        #                 if all(@. abs(C) < 1e-13)
        #                     continue
        #                 end
        #                 push!(rtn, reshape(C, :, OM) \ reshape(D, :, OM))
        #                 flag3 = true
        #                 flag4 = true
        #                 break
        #             end
        #         end
        #         if !flag3
        #             continue
        #         end
        #         break
        #     end
        # end
        # if !flag4
        #     error("Error!")
        # end
        push!(rtn, _3ν(α1, β1, γ))
    end
    println("rank $rank: finished!")
    if isempty(rtn)
        sendbuf = Float64[]
    else
        sendbuf = vcat(vec.(rtn)...)
    end
    if rank == 0
        counts = [sum(@. o ^ 2) for o in OMs]
        recvbuf = zeros(sum(counts))
        MPI.Gatherv!(sendbuf, MPI.VBuffer(recvbuf, counts), 0, comm)
    else
        MPI.Gatherv!(sendbuf, nothing, 0, comm)
    end

    if rank == 0
        cumcounts = vcat(0, cumsum(counts)...)
        for i in 1 : length(names)
            cum2 = vcat(0, cumsum(OMs[i] .^ 2)...)
            for j in 1 : length(names[i])
                table_3ν[names[i][j]] = reshape(recvbuf[cumcounts[i] + 1 : cumcounts[i + 1]][cum2[j] + 1 : cum2[j + 1]], OMs[i][j], OMs[i][j])
            end
        end

        @save "table3nuhalf_SU$(Nc)_$widthmax.jld2" table_3ν
        println("All finished!")
    end

    MPI.Finalize()
end