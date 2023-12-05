struct Block{Nc}
    length::Int
    bonds::Vector{Tuple{Int, Int}}
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list_old::Vector{Int}
    mβ_list::Vector{Int}
    scalar_dict::Dict{Symbol, Vector{Matrix{Float64}}}
end

abstract type EnlargedBlock{Nc} end

struct EnlargedBlockCPU{Nc} <: EnlargedBlock{Nc}
    length::Int
    bonds::Vector{Tuple{Int, Int}}
    α_list::Vector{SUNIrrep{Nc}}
    mα_list::Vector{Int}
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list::Vector{Int}
    mαβ::Matrix{Int}
    scalar_dict::Dict{Symbol, Vector{Matrix{Float64}}}
    tensor_dict::Dict{Int, Matrix{Vector{Matrix{Float64}}}}
end

struct EnlargedBlockGPU{Nc} <: EnlargedBlock{Nc}
    length::Int
    bonds::Vector{Tuple{Int, Int}}
    α_list::Vector{SUNIrrep{Nc}}
    mα_list::Vector{Int}
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list::Vector{Int}
    mαβ::Matrix{Int}
    scalar_dict::Dict{Symbol, Vector{CuMatrix{Float64}}}
    tensor_dict::Dict{Int, Matrix{Vector{Matrix{Float64}}}} # fix later
end

"""
block_enl = enlarge_block(block, block_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = :square)
enlarges the block by one site
"""
function enlarge_block(block::Block{Nc}, block_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = :square) where Nc
    if rank == 0
        αs = copy(block.β_list)
        lenα = length(αs)
        funda = fundamentalirrep(Val(Nc))
        adjoint = adjointirrep(Val(Nc))

        dest = map(α -> unique(keys(directproduct(α, funda))), αs)
        βs = sort!(filter!(β -> on_the_fly || weight(β)[1] <= widthmax, union(dest[block.mβ_list .> 0]...)))
        αβmatrix = [β ∈ d for d in dest, β in βs]
        mαβ = Diagonal(block.mβ_list) * αβmatrix
        @. αβmatrix = mαβ > 0
        lenβ = length(βs)
        cum_mαβ = vcat(zeros(Int, 1, lenβ), cumsum(mαβ, dims = 1))
        ms = cum_mαβ[end, :]
    end

    zy = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(block.length + 1, 2Ly))
    tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
    if rank == 0
        fac1 = sqrt((Nc ^ 2 - 1) / Nc)
        dp = [directproduct(βs[j], adjoint) for j in 1 : lenβ]
        Stemp = map([(i, j) for i in 1 : lenβ, j in 1 : lenβ]) do (i, j)
            o1 = get(dp[j], βs[i], 0)
            map(1 : o1) do τ1
                rtn = zeros(ms[i], ms[j])
                for k in 1 : lenα
                    if αβmatrix[k, i] && αβmatrix[k, j]
                        coeff = on_the_fly ? on_the_fly_calc2(Nc, (αs[k], βs[j], βs[i])) : tables[2][αs[k], βs[j], βs[i]]
                        for l in 1 : mαβ[k, i]
                            rtn[cum_mαβ[k, i] + l, cum_mαβ[k, j] + l] = fac1 * coeff[τ1]
                        end
                    end
                end
                rtn
            end
        end
        MPI.bcast(ms, 0, comm)
        om = length.(Stemp)
        MPI.bcast(om, 0, comm)
    else
        ms = MPI.bcast(nothing, 0, comm)::Vector{Int}
        lenβ = length(ms)
        om = MPI.bcast(nothing, 0, comm)::Matrix{Int}
        Stemp = [[Matrix{Float64}(undef, ms[i], ms[j]) for τ1 in 1 : om[i, j]] for i in 1 : lenβ, j in 1 : lenβ]
    end

    for i in 1 : lenβ, j in 1 : lenβ
        for τ1 in 1 : om[i, j]
            if rank == 0
                MPI.bcast(Stemp[i, j][τ1], 0, comm)
            else
                Stemp[i, j][τ1] .= MPI.bcast(nothing, 0, comm)::Matrix{Float64}
            end
        end
    end

    tensor_dict[zy] = Stemp

    if rank == 0
        y1 = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(block.length, 2Ly))
        y2 = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(block.length + 1, 2Ly))
        zlist = block.length == 0 ? Int[] : ((block.length < Ly || y1 == y2 || (lattice == :honeycombZC && (mod1(block.length + 1, 2Ly) <= Ly ? iseven(y2) : isodd(y2)))) ? [y1] : [y1, y2])
        if (block.length + 1) % Ly == 0
            if y2 == 1
                push!(zlist, Ly)
            else
                push!(zlist, 1)
            end
        end
        fac2 = (Nc ^ 2 - 1) / sqrt(Nc) * signfactor
        bonds = copy(block.bonds)
        Hnew = map(1 : lenβ) do k
            if engine <: GPUEngine
                rtn = any(αβmatrix[:, k]) ? cat(CuArray.(block.scalar_dict[:H][αβmatrix[:, k]])...; dims = (1, 2)) : CUDA.zeros(Float64, 0, 0)
            else
                rtn = any(αβmatrix[:, k]) ? cat(block.scalar_dict[:H][αβmatrix[:, k]]...; dims = (1, 2)) : zeros(0, 0)
            end
            for z in zlist
                for i in 1 : lenα
                    if αβmatrix[i, k]
                        for j in 1 : lenα
                            if αβmatrix[j, k]
                                o1 = length(block_tensor_dict[z][i, j])
                                if o1 > 0
                                    coeff = on_the_fly ? on_the_fly_calc3(Nc, (αs[j], βs[k], αs[i])) : tables[3][αs[j], βs[k], αs[i]]
                                    for τ1 in 1 : o1
                                        temp2 = fac2 * coeff[τ1]
                                        if engine <: GPUEngine
                                            temp3 = CuArray(block_tensor_dict[z][i, j][τ1])
                                        else
                                            temp3 = block_tensor_dict[z][i, j][τ1]
                                        end
                                        @. rtn[cum_mαβ[i, k] + 1 : cum_mαβ[i + 1, k], cum_mαβ[j, k] + 1 : cum_mαβ[j + 1, k]] += temp2 * temp3
                                    end
                                end
                            end
                        end
                    end
                end
            end
            rtn
        end
        if block.length > 0
            push!(bonds, (block.length, block.length + 1))
            if !(block.length < Ly || y1 == y2 || (lattice == :honeycombZC && (mod1(block.length + 1, 2Ly) <= Ly ? iseven(y2) : isodd(y2))))
                push!(bonds, (block.length + 2 - 2mod1(block.length + 1, Ly), block.length + 1))
            end
            if (block.length + 1) % Ly == 0
                push!(bonds, (block.length + 2 - Ly, block.length + 1))
            end
        end
    else
        if engine <: GPUEngine
            Hnew = [CuMatrix{Float64}(undef, m, m) for m in ms]
        else
            Hnew = [Matrix{Float64}(undef, m, m) for m in ms]
        end
    end

    for k in 1 : lenβ
        MPI.Bcast!(Hnew[k], 0, comm)
    end

    if rank == 0
        if engine <: GPUEngine
            block_enl = EnlargedBlockGPU{Nc}(block.length + 1, bonds, αs, copy(block.mβ_list_old), βs, ms, mαβ, Dict{Symbol, Vector{CuMatrix{Float64}}}(:H => Hnew), tensor_dict)
        else
            block_enl = EnlargedBlockCPU{Nc}(block.length + 1, bonds, αs, copy(block.mβ_list_old), βs, ms, mαβ, Dict{Symbol, Vector{Matrix{Float64}}}(:H => Hnew), tensor_dict)
        end
    else
        if engine <: GPUEngine
            block_enl = EnlargedBlockGPU{Nc}(block.length + 1, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], SUNIrrep{Nc}[], Int[], zeros(Int, 0, 0), Dict{Symbol, Vector{CuMatrix{Float64}}}(:H => Hnew), tensor_dict)
        else
            block_enl = EnlargedBlockCPU{Nc}(block.length + 1, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], SUNIrrep{Nc}[], Int[], zeros(Int, 0, 0), Dict{Symbol, Vector{Matrix{Float64}}}(:H => Hnew), tensor_dict)
        end
    end

    block_enl
end

"""
env_tensor_dict = spin_operators!(tensor_table, env, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, fileio, scratch, dirid, block_table, trmat_table, on_the_fly, engine; lattice = :square)
generates spin operators for the environment
"""
function spin_operators!(tensor_table, env::Block{Nc}, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, fileio, scratch, dirid, block_table, trmat_table, on_the_fly, engine; lattice = :square) where Nc
    env_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
    y_conn = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(env.length, 2Ly))
    for y in 1 : min(env.length, Ly)
        if lattice != :honeycombZC || y == y_conn || ((mod1(env.length, 2Ly) <= Ly) ? y == 1 : y == Ly) || (y <= y_conn ? iseven(y) : isodd(y))
            if rank == 0
                if fileio
                    filename = "$scratch/temp$dirid/tensor_$(env_label)_$(env.length)_$y.jld2"
                    if isfile(filename)
                        spin = load_object(filename)::Matrix{Vector{Matrix{Float64}}}
                        rm(filename)
                    else
                        spin = nothing
                    end
                else
                    spin = pop!(tensor_table, (env_label, env.length, y), nothing)
                end

                if isnothing(spin)
                    z = max(2Ly * ((env.length - y) ÷ 2Ly) + y, 2Ly * ((env.length - (1 - y)) ÷ 2Ly) + 1 - y)

                    if fileio
                        env_old = load_object("$scratch/temp$dirid/block_$(env_label)_$(z - 1).jld2")::Block{Nc}
                    else
                        env_old = block_table[env_label, z - 1]
                    end

                    αs = copy(env_old.β_list)
                    lenα = length(αs)
                    funda = fundamentalirrep(Val(Nc))
                    adjoint = adjointirrep(Val(Nc))
            
                    dest = map(α -> unique(keys(directproduct(α, funda))), αs)
                    βs = sort!(filter!(β -> on_the_fly || weight(β)[1] <= widthmax, union(dest[env_old.mβ_list .> 0]...)))
                    αβmatrix = [β ∈ d for d in dest, β in βs]
                    mαβ = Diagonal(env_old.mβ_list) * αβmatrix
                    @. αβmatrix = mαβ > 0
                    lenβ = length(βs)
                    cum_mαβ = vcat(zeros(Int, 1, lenβ), cumsum(mαβ, dims = 1))
                    ms = cum_mαβ[end, :]

                    fac1 = sqrt((Nc ^ 2 - 1) / Nc)
                    dp = [directproduct(βs[j], adjoint) for j in 1 : lenβ]
                    Stemp = map([(i, j) for i in 1 : lenβ, j in 1 : lenβ]) do (i, j)
                        o1 = get(dp[j], βs[i], 0)
                        map(1 : o1) do τ1
                            rtn = zeros(ms[i], ms[j])
                            for k in 1 : lenα
                                if αβmatrix[k, i] && αβmatrix[k, j]
                                    coeff = on_the_fly ? on_the_fly_calc2(Nc, (αs[k], βs[j], βs[i])) : tables[2][αs[k], βs[j], βs[i]]
                                    for l in 1 : mαβ[k, i]
                                        rtn[cum_mαβ[k, i] + l, cum_mαβ[k, j] + l] = fac1 * coeff[τ1]
                                    end
                                end
                            end
                            rtn
                        end
                    end

                    # if engine <: GPUEngine
                    #     Snew = [CUSPARSE.CuSparseMatrixCSC.(Stemp[i, j]) for i in 1 : lenβ, j in 1 : lenβ]
                    # else
                    Snew = Stemp
                    # end

                    if fileio
                        if engine <: GPUEngine
                            env_trmat = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$z.jld2")::Vector{Matrix{Float64}})
                        else
                            env_trmat = load_object("$scratch/temp$dirid/trmat_$(env_label)_$z.jld2")::Vector{Matrix{Float64}}
                        end
                    else
                        if engine <: GPUEngine
                            env_trmat = CuArray.(trmat_table[env_label, z])
                        else
                            env_trmat = trmat_table[env_label, z]
                        end
                    end

                    lennew = length(env_trmat)
                    ms = map(k -> size(env_trmat[k], 2), 1 : lennew)
                    Sold = map(k -> isempty(Snew[k...]) ? Matrix{Float64}[] : [env_trmat[k[1]]' * (M * env_trmat[k[2]]) for M in Snew[k...]], [(ki, kj) for ki in 1 : lennew, kj in 1 : lennew])

                    for x in z + 1 : env.length
                        if engine <: GPUEngine
                            if fileio
                                jldsave("$scratch/temp$dirid/tensor_$(env_label)_$(x - 1)_$y.jld2"; env_tensor_dict = [Array.(Sold[ki, kj]) for ki in 1 : lennew, kj in 1 : lennew])
                            else
                                tensor_table[env_label, x - 1, y] = [Array.(Sold[ki, kj]) for ki in 1 : lennew, kj in 1 : lennew]
                            end
                        else
                            if fileio
                                jldsave("$scratch/temp$dirid/tensor_$(env_label)_$(x - 1)_$y.jld2"; env_tensor_dict = Sold)
                            else
                                tensor_table[env_label, x - 1, y] = deepcopy(Sold)
                            end
                        end

                        αs = copy(βs)
                        lenα = length(αs)
                        dest = map(α -> Set(keys(directproduct(α, funda))), αs)
                        βs = sort!(filter!(β -> on_the_fly || weight(β)[1] <= widthmax, collect(union(dest[ms .> 0]...))))
                        αβmatrix = [β ∈ d for d in dest, β in βs]
                        mαβ = Diagonal(ms) * αβmatrix
                        @. αβmatrix = mαβ > 0
                        lenβ = length(βs)
                        cum_mαβ = vcat(zeros(Int, 1, lenβ), cumsum(mαβ, dims = 1))
                        ms = cum_mαβ[end, :]

                        dp = [directproduct(βs[j], adjoint) for j in 1 : lenβ]
                        dp2 = [directproduct(αs[j], adjoint) for j in 1 : lenα]
                        Snew = map([(i, j) for i in 1 : lenβ, j in 1 : lenβ]) do (i, j)
                            o1 = get(dp[j], βs[i], 0)
                            map(1 : o1) do τ1
                                if engine <: GPUEngine
                                    rtn = CUDA.zeros(Float64, ms[i], ms[j])
                                else
                                    rtn = zeros(ms[i], ms[j])
                                end
                                for ki in findall(αβmatrix[:, i]), kj in findall(αβmatrix[:, j])
                                    if haskey(dp2[kj], αs[ki])
                                        coeff = on_the_fly ? on_the_fly_calc1(Nc, (αs[kj], βs[j], αs[ki], βs[i])) : tables[1][αs[kj], βs[j], αs[ki], βs[i]]
                                        for τ2 in 1 : size(coeff, 1)
                                            @. rtn[cum_mαβ[ki, i] + 1 : cum_mαβ[ki + 1, i], cum_mαβ[kj, j] + 1 : cum_mαβ[kj + 1, j]] += coeff[τ2, τ1] * Sold[ki, kj][τ2]
                                        end
                                    end
                                end
                                rtn
                            end
                        end

                        if fileio
                            if engine <: GPUEngine
                                env_trmat = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$x.jld2")::Vector{Matrix{Float64}})
                            else
                                env_trmat = load_object("$scratch/temp$dirid/trmat_$(env_label)_$x.jld2")::Vector{Matrix{Float64}}
                            end
                        else
                            if engine <: GPUEngine
                                env_trmat = CuArray.(trmat_table[env_label, x])
                            else
                                env_trmat = trmat_table[env_label, x]
                            end
                        end
        
                        lennew = length(env_trmat)
                        ms = map(k -> size(env_trmat[k], 2), 1 : lennew)
                        Sold = map(k -> isempty(Snew[k...]) ? Matrix{Float64}[] : [env_trmat[k[1]]' * (M * env_trmat[k[2]]) for M in Snew[k...]], [(ki, kj) for ki in 1 : lennew, kj in 1 : lennew])
                    end

                    if engine <: GPUEngine
                        spin = [Array.(Sold[ki, kj]) for ki in 1 : lennew, kj in 1 : lennew]
                    else
                        spin = Sold
                    end
                end

                env_tensor_dict[y] = spin
            end
        end
    end
    env_tensor_dict
end