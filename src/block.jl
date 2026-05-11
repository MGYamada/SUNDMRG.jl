# Scalar dictionaries currently carry :H and leave room for additional scalar operators.
const ScalarDictCPU = Dict{Symbol, Vector{Matrix{Float64}}}
const TensorMatrixCPU = Matrix{Vector{Matrix{Float64}}}
const TensorDictCPU = Dict{Int, TensorMatrixCPU}
const ScalarDictGPU = Dict{Symbol, Vector{CuMatrix{Float64}}}
const TensorMatrixGPU = Matrix{Vector{CuMatrix{Float64}}}
const TensorDictGPU = Dict{Int, TensorMatrixGPU}

struct Block{Nc}
    length::Int
    bonds::Vector{Tuple{Int, Int}}
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list_old::Vector{Int}
    mβ_list::Vector{Int}
    scalar_dict::ScalarDictCPU
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
    scalar_dict::ScalarDictCPU
    tensor_dict::TensorDictCPU
end

struct EnlargedBlockGPU{Nc} <: EnlargedBlock{Nc}
    length::Int
    bonds::Vector{Tuple{Int, Int}}
    α_list::Vector{SUNIrrep{Nc}}
    mα_list::Vector{Int}
    β_list::Vector{SUNIrrep{Nc}}
    mβ_list::Vector{Int}
    mαβ::Matrix{Int}
    scalar_dict::ScalarDictGPU
    tensor_dict::TensorDictGPU
end

function _build_spin_tensor(Nc, αs, βs, mαβ, αβmatrix, cum_mαβ, tables, on_the_fly)
    fac = sqrt((Nc ^ 2 - 1) / Nc)
    lenα = length(αs)
    lenβ = length(βs)
    ms = cum_mαβ[end, :]
    adjoint = adjointirrep(Val(Nc))
    dp = [directproduct(βs[j], adjoint) for j in 1 : lenβ]

    map([(i, j) for i in 1 : lenβ, j in 1 : lenβ]) do (i, j)
        o1 = get(dp[j], βs[i], 0)
        map(1 : o1) do τ1
            rtn = zeros(ms[i], ms[j])
            for k in 1 : lenα
                if αβmatrix[k, i] && αβmatrix[k, j]
                    coeff = _on_the_fly(on_the_fly) ? on_the_fly_calc2(Nc, (αs[k], βs[j], βs[i])) : tables[2][αs[k], βs[j], βs[i]]
                    for l in 1 : mαβ[k, i]
                        rtn[cum_mαβ[k, i] + l, cum_mαβ[k, j] + l] = fac * coeff[τ1]
                    end
                end
            end
            rtn
        end
    end
end

function _make_enlarged_block(::Type{<:CPUEngine}, ::Val{Nc}, len, bonds, αs, mαs, βs, ms, mαβ, Hnew, tensor_dict) where Nc
    EnlargedBlockCPU{Nc}(len, bonds, αs, mαs, βs, ms, mαβ, Dict{Symbol, Vector{Matrix{Float64}}}(:H => Hnew), tensor_dict)
end

function _make_enlarged_block(::Type{<:GPUEngine}, ::Val{Nc}, len, bonds, αs, mαs, βs, ms, mαβ, Hnew, tensor_dict) where Nc
    EnlargedBlockGPU{Nc}(len, bonds, αs, mαs, βs, ms, mαβ, Dict{Symbol, Vector{CuMatrix{Float64}}}(:H => Hnew), tensor_dict)
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

        dest = map(α -> unique(keys(directproduct(α, funda))), αs)
        βs = sort!(filter!(β -> _on_the_fly(on_the_fly) || weight(β)[1] <= widthmax, union(dest[block.mβ_list .> 0]...)))
        αβmatrix = [β ∈ d for d in dest, β in βs]
        mαβ = Diagonal(block.mβ_list) * αβmatrix
        @. αβmatrix = mαβ > 0
        lenβ = length(βs)
        cum_mαβ = vcat(zeros(Int, 1, lenβ), cumsum(mαβ, dims = 1))
        ms = cum_mαβ[end, :]
    end

    zy = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(block.length + 1, 2Ly))
    tensor_dict = empty_engine_tensor_dict(engine)
    if rank == 0
        Stemp = _build_spin_tensor(Nc, αs, βs, mαβ, αβmatrix, cum_mαβ, tables, on_the_fly)
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

    tensor_dict[zy] = [to_engine_array.(Ref(engine), Stemp[i, j]) for i in 1 : lenβ, j in 1 : lenβ]

    if rank == 0
        y1 = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(block.length, 2Ly))
        y2 = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(block.length + 1, 2Ly))
        zlist = block.length == 0 ? Int[] : ((block.length < Ly || y1 == y2 || (_is_honeycomb_zc(lattice) && (mod1(block.length + 1, 2Ly) <= Ly ? iseven(y2) : isodd(y2)))) ? [y1] : [y1, y2])
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
            rtn = any(αβmatrix[:, k]) ? cat(to_engine_array.(Ref(engine), block.scalar_dict[:H][αβmatrix[:, k]])...; dims = (1, 2)) : zeros_like_engine(engine, Float64, 0, 0)
            for z in zlist
                for i in 1 : lenα
                    if αβmatrix[i, k]
                        for j in 1 : lenα
                            if αβmatrix[j, k]
                                o1 = length(block_tensor_dict[z][i, j])
                                if o1 > 0
                                    coeff = _on_the_fly(on_the_fly) ? on_the_fly_calc3(Nc, (αs[j], βs[k], αs[i])) : tables[3][αs[j], βs[k], αs[i]]
                                    for τ1 in 1 : o1
                                        temp2 = fac2 * coeff[τ1]
                                        temp3 = block_tensor_dict[z][i, j][τ1]
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
            if !(block.length < Ly || y1 == y2 || (_is_honeycomb_zc(lattice) && (mod1(block.length + 1, 2Ly) <= Ly ? iseven(y2) : isodd(y2))))
                push!(bonds, (block.length + 2 - 2mod1(block.length + 1, Ly), block.length + 1))
            end
            if (block.length + 1) % Ly == 0
                push!(bonds, (block.length + 2 - Ly, block.length + 1))
            end
        end
    else
        Hnew = [engine_matrix_type(engine)(undef, m, m) for m in ms]
    end

    for k in 1 : lenβ
        MPI.Bcast!(Hnew[k], 0, comm)
    end

    if rank != 0
        bonds = Tuple{Int, Int}[]
        αs = typeof(trivialirrep(Val(Nc)))[]
        βs = typeof(trivialirrep(Val(Nc)))[]
        mαβ = zeros(Int, 0, 0)
    end

    _make_enlarged_block(engine, Val(Nc), block.length + 1, bonds, αs, rank == 0 ? copy(block.mβ_list_old) : Int[], βs, ms, mαβ, Hnew, tensor_dict)
end

"""
env_tensor_dict = spin_operators!(storage, env, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = :square)
generates spin operators for the environment
"""
function spin_operators!(storage::AbstractInternalStorage, env::Block{Nc}, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = :square) where Nc
    env_tensor_dict = empty_engine_tensor_dict(engine)
    y_conn = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(env.length, 2Ly))
    for y in 1 : min(env.length, Ly)
        if !_is_honeycomb_zc(lattice) || y == y_conn || ((mod1(env.length, 2Ly) <= Ly) ? y == 1 : y == Ly) || (y <= y_conn ? iseven(y) : isodd(y))
            if rank == 0
                if has_tensor(storage, env_label, env.length, y)
                    spin = take_tensor!(storage, env_label, env.length, y)
                else
                    z = max(2Ly * ((env.length - y) ÷ 2Ly) + y, 2Ly * ((env.length - (1 - y)) ÷ 2Ly) + 1 - y)

                    env_old = load_block(storage, env_label, z - 1)::Block{Nc}

                    αs = copy(env_old.β_list)
                    lenα = length(αs)
                    funda = fundamentalirrep(Val(Nc))
                    adjoint = adjointirrep(Val(Nc))
            
                    dest = map(α -> unique(keys(directproduct(α, funda))), αs)
                    βs = sort!(filter!(β -> _on_the_fly(on_the_fly) || weight(β)[1] <= widthmax, union(dest[env_old.mβ_list .> 0]...)))
                    αβmatrix = [β ∈ d for d in dest, β in βs]
                    mαβ = Diagonal(env_old.mβ_list) * αβmatrix
                    @. αβmatrix = mαβ > 0
                    lenβ = length(βs)
                    cum_mαβ = vcat(zeros(Int, 1, lenβ), cumsum(mαβ, dims = 1))
                    ms = cum_mαβ[end, :]

                    Stemp = _build_spin_tensor(Nc, αs, βs, mαβ, αβmatrix, cum_mαβ, tables, on_the_fly)

                    Snew = [to_engine_array.(Ref(engine), Stemp[i, j]) for i in 1 : lenβ, j in 1 : lenβ]

                    env_trmat = load_trmat(storage, env_label, z, engine)

                    lennew = length(env_trmat)
                    ms = map(k -> size(env_trmat[k], 2), 1 : lennew)
                    Sold = map(k -> isempty(Snew[k...]) ? empty_engine_matrix_vector(engine) : [env_trmat[k[1]]' * (M * env_trmat[k[2]]) for M in Snew[k...]], [(ki, kj) for ki in 1 : lennew, kj in 1 : lennew])

                    for x in z + 1 : env.length
                        save_tensor(storage, env_label, x - 1, y, host_tensor_for_save(engine, Sold, lennew))

                        αs = copy(βs)
                        lenα = length(αs)
                        dest = map(α -> Set(keys(directproduct(α, funda))), αs)
                        βs = sort!(filter!(β -> _on_the_fly(on_the_fly) || weight(β)[1] <= widthmax, collect(union(dest[ms .> 0]...))))
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
                                rtn = zeros_like_engine(engine, Float64, ms[i], ms[j])
                                for ki in findall(αβmatrix[:, i]), kj in findall(αβmatrix[:, j])
                                    if haskey(dp2[kj], αs[ki])
                                        coeff = _on_the_fly(on_the_fly) ? on_the_fly_calc1(Nc, (αs[kj], βs[j], αs[ki], βs[i])) : tables[1][αs[kj], βs[j], αs[ki], βs[i]]
                                        for τ2 in 1 : size(coeff, 1)
                                            @. rtn[cum_mαβ[ki, i] + 1 : cum_mαβ[ki + 1, i], cum_mαβ[kj, j] + 1 : cum_mαβ[kj + 1, j]] += coeff[τ2, τ1] * Sold[ki, kj][τ2]
                                        end
                                    end
                                end
                                rtn
                            end
                        end

                        env_trmat = load_trmat(storage, env_label, x, engine)
        
                        lennew = length(env_trmat)
                        ms = map(k -> size(env_trmat[k], 2), 1 : lennew)
                        Sold = map(k -> isempty(Snew[k...]) ? empty_engine_matrix_vector(engine) : [env_trmat[k[1]]' * (M * env_trmat[k[2]]) for M in Snew[k...]], [(ki, kj) for ki in 1 : lennew, kj in 1 : lennew])
                    end

                    spin = host_tensor_for_save(engine, Sold, lennew)
                end

                env_tensor_dict[y] = spin
            end
        end
    end
    env_tensor_dict
end
