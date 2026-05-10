function _build_block_enlarging(Nc, αs, βs, dp, αβmatrix, cum_mαβ, adjoint, tables, on_the_fly)
    dp2 = map(x -> directproduct(x, adjoint), αs)
    enlarging = BlockEnlarging[]
    len = length(βs)
    for i in 1 : len, j in 1 : len
        if haskey(dp[j], βs[i])
            for ki in findall(αβmatrix[:, i]), kj in findall(αβmatrix[:, j])
                if haskey(dp2[kj], αs[ki])
                    matrix = on_the_fly ? on_the_fly_calc1(Nc, (αs[kj], βs[j], αs[ki], βs[i])) : tables[1][αs[kj], βs[j], αs[ki], βs[i]]
                    for τ1 in 1 : size(matrix, 2), τ2 in 1 : size(matrix, 1)
                        push!(enlarging, BlockEnlarging(matrix[τ2, τ1], τ1, τ2, cum_mαβ[ki, i] + 1 : cum_mαβ[ki + 1, i], cum_mαβ[kj, j] + 1 : cum_mαβ[kj + 1, j], i, j, ki, kj))
                    end
                end
            end
        end
    end
    enlarging
end

function _empty_tensor_hold(engine)
    empty_engine_tensor_dict(engine)
end

function _broadcast_tensor_holds(tensor_dict, mβ_list, conn, held_sites, comm, rank, engine; to_engine = false)
    tensor_dict_hold = _empty_tensor_hold(engine)

    if rank == 0
        site_list = MPI.bcast(collect(keys(tensor_dict)), 0, comm)::Vector{Int}
    else
        site_list = MPI.bcast(nothing, 0, comm)::Vector{Int}
    end

    for site in site_list
        if site != conn
            if rank == 0
                Sold = tensor_dict[site]
                mms = MPI.bcast(mβ_list, 0, comm)::Vector{Int}
                len = length(mms)
                om = length.(Sold)
                MPI.bcast(om, 0, comm)
            else
                mms = MPI.bcast(nothing, 0, comm)::Vector{Int}
                len = length(mms)
                om = MPI.bcast(nothing, 0, comm)::Matrix{Int}
                Sold = [[Matrix{Float64}(undef, mms[i], mms[j]) for τ1 in 1 : om[i, j]] for i in 1 : len, j in 1 : len]
            end

            for i in 1 : len, j in 1 : len
                for τ1 in 1 : om[i, j]
                    MPI.Bcast!(Sold[i, j][τ1], 0, comm)
                end
            end

            if site ∈ held_sites
                if to_engine
                    tensor_dict_hold[site] = [to_engine_array.(Ref(engine), Sold[i, j]) for i in 1 : len, j in 1 : len]
                else
                    tensor_dict_hold[site] = Sold
                end
            end
        end
    end

    tensor_dict_hold
end

function _step_bonds(sys_length, env_length, Ly, lattice)
    superblock_bonds = [(y, y) for y in 1 : min(sys_length, env_length, Ly)]
    x_conn = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(sys_length, 2Ly))
    y_conn = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(env_length, 2Ly))

    if lattice == :honeycombZC
        filter!(z -> z[1] <= min(x_conn, y_conn) ? iseven(z[1]) : isodd(z[1]), superblock_bonds)
    end
    if (x_conn, y_conn) ∉ superblock_bonds && !(lattice == :honeycombZC && sys_length <= Ly && env_length <= Ly)
        push!(superblock_bonds, (x_conn, y_conn))
    end

    sort!(superblock_bonds)
    if (sys_length >= Ly || env_length >= Ly) && sys_length % Ly != 0
        if mod1(sys_length, 2Ly) <= Ly
            push!(superblock_bonds, (1, Ly))
        else
            push!(superblock_bonds, (Ly, 1))
        end
    end

    return superblock_bonds, x_conn, y_conn
end

function _rank_held_bonds(superblock_bonds, Ncpu, rank)
    dc = dcinit(length(superblock_bonds), Ncpu)
    bonds_hold = superblock_bonds[dc[rank + 1] + 1 : dc[rank + 2]]
    holdmax = maximum(dc[2 : end] .- dc[1 : end - 1])
    while length(bonds_hold) < holdmax
        push!(bonds_hold, (0, 0))
    end
    return bonds_hold
end

function _build_superblock_h1(sys_enl, env_enl, OM, Ncpu, rank, engine)
    if engine <: GPUEngine
        superblock_H1 = SuperBlock1GPU[]
    else
        superblock_H1 = SuperBlock1CPU[]
    end

    for k1 in axes(OM, 1), k2 in axes(OM, 2)
        if OM[k1, k2] > 0 && (k1 + k2 - 2) % Ncpu == rank
            if engine <: GPUEngine
                push!(superblock_H1, SuperBlock1GPU(sys_enl.scalar_dict[:H][k1], env_enl.scalar_dict[:H][k2], k1, k2))
            else
                push!(superblock_H1, SuperBlock1CPU(sys_enl.scalar_dict[:H][k1], env_enl.scalar_dict[:H][k2], k1, k2))
            end
        end
    end

    return superblock_H1
end

function _build_superblock_h2(Nc, sys_βs, env_βs, sys_ms, env_ms, sys_dp, env_dp, OM, γirrep, signfactor, tables, on_the_fly)
    fac = sqrt(Nc ^ 2 - 1) * signfactor
    superblock_H2 = SuperBlock2[]

    for k1 in eachindex(sys_βs), k2 in eachindex(env_βs)
        for om1 in 1 : OM[k1, k2]
            miniblock = MiniBlock[]
            for k3 in eachindex(sys_βs)
                if haskey(sys_dp[k1], sys_βs[k3])
                    for k4 in eachindex(env_βs)
                        if haskey(env_dp[k2], env_βs[k4]) && OM[k3, k4] > 0
                            ctensor = on_the_fly ? on_the_fly_calc4(Nc, (sys_βs[k1], env_βs[k2], γirrep, sys_βs[k3], env_βs[k4])) : tables[4][sys_βs[k1], env_βs[k2], γirrep, sys_βs[k3], env_βs[k4]]
                            for τ1 in 1 : size(ctensor, 1), τ2 in 1 : size(ctensor, 2), om2 in 1 : size(ctensor, 3)
                                push!(miniblock, MiniBlock(fac * ctensor[τ1, τ2, om2, om1], om2, τ1, τ2, k3, k4))
                            end
                        end
                    end
                end
            end
            push!(superblock_H2, SuperBlock2(miniblock, om1, k1, k2, sys_ms[k1], env_ms[k2]))
        end
    end

    return superblock_H2
end

function _connection_tensor(block_enl, conn, held_sites)
    if conn ∈ held_sites
        return block_enl.tensor_dict[conn]
    end
    return nothing
end

function _prepare_step_workspace(sys, env, sys_tensor_dict, env_tensor_dict, sys_enl, env_enl, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, lattice, ::Val{Nc}) where Nc
    sys_αs = MPI.bcast(sys_enl.α_list, 0, comm)::Vector{<:SUNIrrep{Nc}}
    env_αs = MPI.bcast(env_enl.α_list, 0, comm)::Vector{<:SUNIrrep{Nc}}
    sys_βs = MPI.bcast(sys_enl.β_list, 0, comm)::Vector{<:SUNIrrep{Nc}}
    env_βs = MPI.bcast(env_enl.β_list, 0, comm)::Vector{<:SUNIrrep{Nc}}
    sys_ms = MPI.bcast(sys_enl.mβ_list, 0, comm)::Vector{Int}
    env_ms = MPI.bcast(env_enl.mβ_list, 0, comm)::Vector{Int}
    sys_mαβ = MPI.bcast(sys_enl.mαβ, 0, comm)::Matrix{Int}
    env_mαβ = MPI.bcast(env_enl.mαβ, 0, comm)::Matrix{Int}

    sys_len = length(sys_βs)
    env_len = length(env_βs)
    sys_αβmatrix = sys_mαβ .> 0
    env_αβmatrix = env_mαβ .> 0
    cum_sys_mαβ = vcat(zeros(Int, 1, sys_len), cumsum(sys_mαβ, dims = 1))
    cum_env_mαβ = vcat(zeros(Int, 1, env_len), cumsum(env_mαβ, dims = 1))

    γirrep = γ_list[mod1(sys_enl.length + env_enl.length, Nc)]
    adjoint = adjointirrep(Val(Nc))

    superblock_bonds, x_conn, y_conn = _step_bonds(sys_enl.length, env_enl.length, Ly, lattice)
    bonds_hold = _rank_held_bonds(superblock_bonds, Ncpu, rank)
    held_x = first.(bonds_hold)
    held_y = last.(bonds_hold)

    sys_dp = map(x -> directproduct(x, adjoint), sys_βs)
    env_dp = map(x -> directproduct(x, adjoint), env_βs)
    sys_enlarge = _build_block_enlarging(Nc, sys_αs, sys_βs, sys_dp, sys_αβmatrix, cum_sys_mαβ, adjoint, tables, on_the_fly)
    env_enlarge = _build_block_enlarging(Nc, env_αs, env_βs, env_dp, env_αβmatrix, cum_env_mαβ, adjoint, tables, on_the_fly)

    OM = OM_matrix(sys_βs, env_βs, γirrep)
    superblock_H1 = _build_superblock_h1(sys_enl, env_enl, OM, Ncpu, rank, engine)
    superblock_H2 = _build_superblock_h2(Nc, sys_βs, env_βs, sys_ms, env_ms, sys_dp, env_dp, OM, γirrep, signfactor, tables, on_the_fly)

    sys_connS = _connection_tensor(sys_enl, x_conn, held_x)
    env_connS = _connection_tensor(env_enl, y_conn, held_y)
    sys_tensor_dict_hold = _broadcast_tensor_holds(sys_tensor_dict, sys.mβ_list, x_conn, held_x, comm, rank, engine; to_engine = engine <: GPUEngine)
    env_tensor_dict_hold = _broadcast_tensor_holds(env_tensor_dict, env.mβ_list, y_conn, held_y, comm, rank, engine)

    return _StepWorkspace(sys_αs, env_αs, sys_βs, env_βs, sys_ms, env_ms, sys_len, env_len, superblock_bonds, bonds_hold, x_conn, y_conn, sys_dp, env_dp, sys_enlarge, env_enlarge, OM, superblock_H1, superblock_H2, sys_connS, env_connS, sys_tensor_dict_hold, env_tensor_dict_hold)
end

function _initial_step_wavefunction(Ψ0_guess, env_ms, sys_ms, OM, Ncpu, rank, engine)
    if !isnothing(Ψ0_guess)
        return deepcopy(Ψ0_guess)
    end

    if engine <: GPUEngine
        return [[CUDA.rand(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : ((kj + ki - 2) % Ncpu == rank ? OM[kj, ki] : 0)] for ki in eachindex(env_ms), kj in eachindex(sys_ms)]
    end

    return [[rand(env_ms[ki], sys_ms[kj]) for J in 1 : ((kj + ki - 2) % Ncpu == rank ? OM[kj, ki] : 0)] for ki in eachindex(env_ms), kj in eachindex(sys_ms)]
end

function _step_result_buffers(::Val{Nc}, engine) where Nc
    newblock = Block{Nc}[]
    if engine <: GPUEngine
        newblock_enl = EnlargedBlockGPU{Nc}[]
        trmat = Vector{CuMatrix{Float64}}[]
        newtensor_dict = Dict{Int64, Matrix{Vector{CuMatrix{Float64}}}}[]
    else
        newblock_enl = EnlargedBlockCPU{Nc}[]
        trmat = Vector{Matrix{Float64}}[]
        newtensor_dict = Dict{Int64, Matrix{Vector{Matrix{Float64}}}}[]
    end

    return newblock, newblock_enl, trmat, newtensor_dict
end

function _step_side_context(switch, sys_label, sys_len, env_len, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, x_conn, y_conn, sys, env, sys_enl, env_enl)
    env_label = sys_label == :l ? :r : :l

    if switch == 1
        return _StepSideContext(
            sys_len,
            sys_ms,
            sys_βs,
            sys_dp,
            x_conn,
            sys_label,
            sys,
            sys_enl,
        )
    end

    return _StepSideContext(
        env_len,
        env_ms,
        env_βs,
        env_dp,
        y_conn,
        env_label,
        env,
        env_enl,
    )
end
