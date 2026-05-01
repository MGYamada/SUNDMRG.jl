struct SuperBlock1CPU
    sys_H::Matrix{Float64}
    env_H::Matrix{Float64}
    sys_ind::Int
    env_ind::Int
end

struct SuperBlock1GPU
    sys_H::CuMatrix{Float64}
    env_H::CuMatrix{Float64}
    sys_ind::Int
    env_ind::Int
end

struct MiniBlock
    coeff::Float64
    om2::Int
    sys_τ::Int
    env_τ::Int
    sys_out::Int
    env_out::Int
end

struct SuperBlock2
    miniblock::Vector{MiniBlock}
    om1::Int
    sys_in::Int
    env_in::Int
    sys_in_size::Int
    env_in_size::Int
end

struct BlockEnlarging
    coeff::Float64
    τ1::Int
    τ2::Int
    range_i::UnitRange
    range_j::UnitRange
    i::Int
    j::Int
    ki::Int
    kj::Int
end

struct _StepSideContext
    len::Int
    ms
    betas
    dp
    conn::Int
    label::Symbol
    block
    block_enl
end

struct _StepLanczosContext
    target::Int
    comm
    rank::Int
    engine
    alg::Symbol
    superblock_H1
    bonds_hold
    x_conn::Int
    y_conn::Int
    sys_connS
    env_connS
    sys_ms
    env_ms
    sys_βs
    env_βs
    sys_dp
    env_dp
    sys_enlarge
    env_enlarge
    sys_tensor_dict_hold
    env_tensor_dict_hold
    superblock_H2
    OM
    sys_len::Int
    env_len::Int
    Ncpu::Int
end

struct _StepDensityContext
    comm
    rank::Int
    Ncpu::Int
    engine
    α::Float64
    m::Int
    ES_max::Float64
    noisy::Bool
end

struct _StepMeasurementContext
    SiSj
    Ly::Int
    x_conn::Int
    y_conn::Int
    sys_connS
    env_connS
    sys_len::Int
    env_len::Int
    sys_tensor_dict
    env_tensor_dict
    sys_tensor_dict_hold
    env_tensor_dict_hold
    sys_αs
    env_αs
    sys_βs
    env_βs
    sys_dp
    env_dp
    sys_ms
    env_ms
    sys_enlarge
    env_enlarge
    superblock_H2
    OM
    comm
    rank::Int
    Ncpu::Int
    engine
    correlation::Symbol
    margin::Int
    lattice::Symbol
end

struct _StepBlockContext
    Ly::Int
    widthmax::Int
    signfactor::Float64
    comm
    rank::Int
    Ncpu::Int
    tables
    on_the_fly::Bool
    engine
    lattice::Symbol
end

struct _StepCorrectionContext
    superblock_bonds
    sys_connS
    env_connS
    sys_enl
    env_enl
    x_conn::Int
    y_conn::Int
    sys_tensor_dict_hold
    env_tensor_dict_hold
    sys_tensor_dict
    env_tensor_dict
    sys_ms
    env_ms
    sys_dp
    env_dp
    sys_βs
    env_βs
    sys_enlarge
    env_enlarge
    engine
end

struct _StepWorkspace
    sys_αs
    env_αs
    sys_βs
    env_βs
    sys_ms
    env_ms
    sys_len::Int
    env_len::Int
    superblock_bonds
    bonds_hold
    x_conn::Int
    y_conn::Int
    sys_dp
    env_dp
    sys_enlarge
    env_enlarge
    OM
    superblock_H1
    superblock_H2
    sys_connS
    env_connS
    sys_tensor_dict_hold
    env_tensor_dict_hold
end

"""
dc = dcinit(N, Ncpu)
Divide conquer
"""
function dcinit(N, Ncpu)
    @. N * [0 : Ncpu;] ÷ Ncpu
end

"""
Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
Function barrier
"""
function Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
    Ψtemp = [[zeros_like_engine(engine, Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]

    for s in superblock_H2
        root1 = (s.sys_in + s.env_in - 2) % Ncpu
        if rank == root1
            temp3 = Ψin[s.env_in, s.sys_in][s.om1]
        else
            temp3 = engine_matrix_type(engine)(undef, s.env_in_size, s.sys_in_size)
        end
        if engine <: GPUEngine
            CUDA.synchronize()
        end
        MPI.Bcast!(temp3, root1, comm)

        if (x, y) != (0, 0)
            for m in s.miniblock
                temp4 = env_S[m.env_out, s.env_in][m.env_τ] * (sys_S[m.sys_out, s.sys_in][m.sys_τ] * temp3')'
                @. Ψtemp[m.env_out, m.sys_out][m.om2] += m.coeff * temp4
            end
        end
    end

    for ki in 1 : env_len, kj in 1 : sys_len
        for om in 1 : OM[kj, ki]
            root2 = (kj + ki - 2) % Ncpu
            if engine <: GPUEngine
                CUDA.synchronize()
            end
            MPI.Reduce!(Ψtemp[ki, kj][om], MPI.SUM, root2, comm)
            if rank == root2
                Ψout[ki, kj][om] .+= Ψtemp[ki, kj][om]
            end
        end
    end
end

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
    if engine <: GPUEngine
        Dict{Int, Matrix{Vector{CuMatrix{Float64}}}}()
    else
        Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
    end
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

function _density_matrix_balancer(ms, Ncpu, rank)
    len = length(ms)
    balancer = zeros(Int, len)
    if rank == 0 && Ncpu > 1 && len > 0
        max_m = maximum(ms)
        loads = @. (ms / max_m) ^ 3
        load_sum = zeros(Float64, Ncpu)
        load_sum[1] = sum(loads)
        load_max = maximum(load_sum)
        for i in 1 : 100
            j = rand(1 : len)
            new_b = rand(filter(x -> x != balancer[j], 0 : Ncpu - 1))
            load_sum[balancer[j] + 1] -= loads[j]
            load_sum[new_b + 1] += loads[j]
            new_load_max = maximum(load_sum)
            if log(rand()) < 100.0 * (load_max - new_load_max)
                balancer[j] = new_b
                load_max = new_load_max
            else
                load_sum[balancer[j] + 1] += loads[j]
                load_sum[new_b + 1] -= loads[j]
            end
        end
    end
    return balancer
end

function _empty_density_matrix_vector(engine)
    if engine <: GPUEngine
        return CuMatrix{Float64}[]
    end
    return Matrix{Float64}[]
end

function _reduced_density_matrices(Ψ0, switch, side::_StepSideContext, dimβ, sys_len, env_len, context::_StepDensityContext)
    (; len, ms) = side
    (; comm, rank, engine) = context
    ρs = _empty_density_matrix_vector(engine)

    if switch == 1
        for k in 1 : len
            fac = 1.0 / dimβ[k]
            if engine <: GPUEngine
                ρ = zeros_like_engine(engine, Float64, ms[k], ms[k])
                for j in 1 : env_len
                    for J in eachindex(Ψ0[j, k])
                        CUBLAS.syrk!('U', 'T', fac, Ψ0[j, k][J], 1.0, ρ)
                    end
                end
                CUDA.synchronize()
            else
                ρ = zeros_like_engine(engine, Float64, ms[k], ms[k])
                for j in 1 : env_len
                    for J in eachindex(Ψ0[j, k])
                        BLAS.syrk!('U', 'T', fac, Ψ0[j, k][J], 1.0, ρ)
                    end
                end
            end
            MPI.Reduce!(ρ, MPI.SUM, 0, comm)
            if rank == 0
                push!(ρs, ρ)
            end
        end
    else
        for k in 1 : len
            fac = 1.0 / dimβ[k]
            if engine <: GPUEngine
                ρ = zeros_like_engine(engine, Float64, ms[k], ms[k])
                for j in 1 : sys_len
                    for J in eachindex(Ψ0[k, j])
                        CUBLAS.syrk!('U', 'N', fac, Ψ0[k, j][J], 1.0, ρ)
                    end
                end
                CUDA.synchronize()
            else
                ρ = zeros_like_engine(engine, Float64, ms[k], ms[k])
                for j in 1 : sys_len
                    for J in eachindex(Ψ0[k, j])
                        BLAS.syrk!('U', 'N', fac, Ψ0[k, j][J], 1.0, ρ)
                    end
                end
            end
            MPI.Reduce!(ρ, MPI.SUM, 0, comm)
            if rank == 0
                push!(ρs, ρ)
            end
        end
    end

    return ρs
end

function _balanced_density_matrices(ρs, balancer, side::_StepSideContext, context::_StepDensityContext)
    (; len, ms) = side
    (; comm, rank, engine) = context
    if rank == 0
        ρnew = ρs[balancer .== 0]
    else
        ρnew = _empty_density_matrix_vector(engine)
    end

    for k in 1 : len
        if balancer[k] != 0
            if rank == 0
                MPI.Send(ρs[k], balancer[k], k, comm)
            elseif rank == balancer[k]
                ρ = engine_matrix_type(engine)(undef, ms[k], ms[k])
                MPI.Recv!(ρ, 0, k, comm)
                push!(ρnew, ρ)
            end
        end
    end

    return ρnew
end

function _density_eigendecomposition(ρnew, context::_StepDensityContext)
    (; engine) = context
    local λζtemp
    time_DM = @elapsed begin
        if engine <: GPUEngine
            λζtemp = map(ρ -> magma_syevd!('V', 'U', ρ), ρnew)
        else
            λζtemp = map(ρ -> LAPACK.syev!('V', 'U', ρ), ρnew)
        end
    end

    return λζtemp, time_DM
end

function _collect_density_eigenpairs(λζtemp, balancer, side::_StepSideContext, context::_StepDensityContext)
    (; len, ms) = side
    (; comm, rank) = context
    if rank == 0
        λ = [Vector{Float64}(undef, ms[k]) for k in 1 : len]
        ζ = [Matrix{Float64}(undef, ms[k], ms[k]) for k in 1 : len]
        @. λ[balancer .== 0] = first(λζtemp)
        @. ζ[balancer .== 0] = last(λζtemp)
    else
        λ = Vector{Float64}[]
        ζ = Matrix{Float64}[]
    end

    for k in 1 : len
        if balancer[k] != 0
            if rank == balancer[k]
                pos = count(balancer[1 : k] .== rank)
                MPI.Send(λζtemp[pos][1], 0, k, comm)
                MPI.Send(λζtemp[pos][2], 0, k + len, comm)
            elseif rank == 0
                MPI.Recv!(λ[k], balancer[k], k, comm)
                MPI.Recv!(ζ[k], balancer[k], k + len, comm)
            end
        end
    end

    return λ, ζ
end

function _density_truncation_basis(λ, ζ, side::_StepSideContext, dimβ, sys_enl, env_enl, context::_StepDensityContext, ::Val{Nc}, switch) where Nc
    (; m, ES_max, engine, rank, noisy) = context
    βs = side.betas
    block_enl = side.block_enl
    esi = Dict{SUNIrrep{Nc}, Vector{Float64}}()
    if rank == 0
        ee = -sum(@. dimβ * sum(map(x -> x * log(x), filter(x -> x > 0.0, λ))))
        if ES_max != -Inf && sys_enl.length == env_enl.length
            for (β, squared) in zip(βs, λ)
                thre = exp(-ES_max)
                filtered = reverse!(filter(x -> x >= thre, squared))
                if !isempty(filtered)
                    esi[β] = @. -log(filtered)
                end
            end
        end
        Λ = sort!(collect(Iterators.flatten(reverse.(λ))), rev = true)
        λthreshold = m < length(Λ) ? Λ[m + 1] : -Inf
        indices = map(x -> x .> λthreshold, λ)
        transformation_matrix = map(x -> to_engine_array(engine, x[1][:, x[2]]), zip(ζ, indices))

        msnew = map(x -> size(x, 2), transformation_matrix)
        if noisy && switch == 1
            println("Keeping ", sum(msnew), " SU($Nc) states equivalent to ", sum(dimβ .* msnew), " U(1) states")
        end
        Hnew = map(x -> Array(x[2]' * (x[1] * x[2])), zip(block_enl.scalar_dict[:H], transformation_matrix))
        return ee, esi, transformation_matrix, msnew, Hnew, indices
    end

    transformation_matrix = [engine_matrix_type(engine)(undef, 0, 0)]
    return 0.0, esi, transformation_matrix, Int[], Matrix{Float64}[], Vector{Bool}[]
end

function _apply_density_matrix_correction!(ρs, switch, side::_StepSideContext, density_context::_StepDensityContext, correction_context::_StepCorrectionContext)
    (; α) = density_context
    (; superblock_bonds, sys_connS, env_connS, sys_enl, env_enl, x_conn, y_conn, sys_tensor_dict_hold, env_tensor_dict_hold, sys_tensor_dict, env_tensor_dict, sys_ms, env_ms, sys_dp, env_dp, sys_βs, env_βs, sys_enlarge, env_enlarge, engine) = correction_context
    (; len, ms, dp, conn) = side
    βs = side.betas
    Sρs = [zeros_like_engine(engine, Float64, ms[k], ms[k]) for k in 1 : len]

    for x in unique(map(z -> z[switch], superblock_bonds))
        if x == conn
            if switch == 1
                Stemp = isnothing(sys_connS) ? sys_enl.tensor_dict[x_conn] : sys_connS
            else
                Stemp = isnothing(env_connS) ? env_enl.tensor_dict[y_conn] : env_connS
            end
        else
            if switch == 1
                Sx = haskey(sys_tensor_dict_hold, x) ? sys_tensor_dict_hold[x] : sys_tensor_dict[x]
                Stemp = [[zeros_like_engine(engine, Float64, sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in eachindex(sys_ms), j in eachindex(sys_ms)]
                for e in sys_enlarge
                    @. Stemp[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sx[e.ki, e.kj][e.τ2]
                end
            else
                Sx = haskey(env_tensor_dict_hold, x) ? env_tensor_dict_hold[x] : env_tensor_dict[x]
                Stemp = [[zeros_like_engine(engine, Float64, env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in eachindex(env_ms), j in eachindex(env_ms)]
                for e in env_enlarge
                    @. Stemp[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sx[e.ki, e.kj][e.τ2]
                end
            end
        end

        for k in 1 : len, l in 1 : len
            if haskey(dp[k], βs[l])
                for τ1 in 1 : length(Stemp[l, k])
                    if engine <: GPUEngine && Stemp[l, k][τ1] isa CuMatrix{Float64}
                        Sρs[l] .+= UpperTriangular(Stemp[l, k][τ1] * CUBLAS.symm('R', 'U', ρs[k], Stemp[l, k][τ1])')
                    else
                        Sρs[l] .+= UpperTriangular(Stemp[l, k][τ1] * (Stemp[l, k][τ1] * Symmetric(ρs[k]))')
                    end
                end
            end
        end
    end

    for k in 1 : len
        @. ρs[k] += α * Sρs[k]
    end

    return ρs
end

function _measure_step_tensor_dict!(context::_StepMeasurementContext, switch, label, sys, env, sys_enl, env_enl, Ψ0, transformation_matrix, Sj)
    (; SiSj, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, comm, rank, Ncpu, engine, correlation, margin, lattice) = context

    if switch == 1
        return measurement!(SiSj, label, sys, env, sys_enl, env_enl, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = correlation, margin = margin, lattice = lattice, Sj = Sj)
    end

    # Strictly speaking, superblock_H2 is not exchanged here, but don't worry. It works.
    return measurement!(SiSj, label, env, sys, env_enl, sys_enl, Ly, y_conn, x_conn, env_connS, sys_connS, env_len, sys_len, env_tensor_dict, sys_tensor_dict, env_tensor_dict_hold, sys_tensor_dict_hold, env_αs, sys_αs, env_βs, sys_βs, env_dp, sys_dp, env_ms, sys_ms, env_enlarge, sys_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = correlation, margin = margin, lattice = lattice, Sj = Sj)
end

function _push_step_block!(newblock, newtensor_dict, newblock_enl, tensor_dict, side::_StepSideContext, msnew, Hnew, context::_StepBlockContext, ::Val{Nc}) where Nc
    (; block_enl) = side
    (; Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine, lattice) = context

    if rank == 0
        push!(newblock, Block(block_enl.length, block_enl.bonds, block_enl.β_list, block_enl.mβ_list, msnew, Dict{Symbol, Vector{Matrix{Float64}}}(:H => Hnew)))
    else
        push!(newblock, Block(block_enl.length, Tuple{Int, Int}[], typeof(trivialirrep(Val(Nc)))[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}()))
    end

    push!(newtensor_dict, tensor_dict)
    push!(newblock_enl, enlarge_block(newblock[end], tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice))
    return nothing
end

function _step_truncation_error(λ, indices, dimβ, block_enl, ::Val{Nc}) where Nc
    λnew = map(x -> x[1][x[2]], zip(λ, indices))
    kept_weight = sum(map(x -> sum(sort(sum.(λnew[block_enl.mαβ[:, x[1]] .> 0]); by = abs)) * dim(x[2]), enumerate(block_enl.β_list)))
    return sum(sort(@. sum(λ) * dimβ; by = abs)) - kept_weight / Nc
end

function _step_spin_operator(x, conn, connS, ms, βs, dp, enlarging, tensor_dict_hold, engine)
    if x == conn
        return connS
    elseif x > 0
        spin = [[zeros_like_engine(engine, Float64, ms[i], ms[j]) for τ1 in 1 : get(dp[j], βs[i], 0)] for i in eachindex(ms), j in eachindex(ms)]
        for e in enlarging
            @. spin[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * tensor_dict_hold[x][e.ki, e.kj][e.τ2]
        end
        return spin
    end

    return nothing
end

function _run_step_lanczos!(Ψ0, context::_StepLanczosContext)
    (; target, comm, rank, engine, alg, superblock_H1, bonds_hold, x_conn, y_conn, sys_connS, env_connS, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, sys_enlarge, env_enlarge, sys_tensor_dict_hold, env_tensor_dict_hold, superblock_H2, OM, sys_len, env_len, Ncpu) = context

    local E
    time_Lanczos = @elapsed E = Lanczos!(Ψ0, target + 1, comm, rank, engine; alg = alg) do Ψout, Ψin
        for s in superblock_H1
            for J in eachindex(Ψin[s.env_ind, s.sys_ind])
                temp1 = Ψin[s.env_ind, s.sys_ind][J] * s.sys_H'
                temp2 = s.env_H * Ψin[s.env_ind, s.sys_ind][J]
                @. Ψout[s.env_ind, s.sys_ind][J] = temp1 + temp2
            end
        end

        for (x, y) in bonds_hold
            sys_S = _step_spin_operator(x, x_conn, sys_connS, sys_ms, sys_βs, sys_dp, sys_enlarge, sys_tensor_dict_hold, engine)
            env_S = _step_spin_operator(y, y_conn, env_connS, env_ms, env_βs, env_dp, env_enlarge, env_tensor_dict_hold, engine)
            Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
        end
    end

    return E, time_Lanczos
end

function _step_energy(E, time_Lanczos, sys_enl, env_enl, superblock_bonds, comm, rank, noisy, ::Val{Nc}) where Nc
    time1 = MPI.Reduce(time_Lanczos, MPI.MAX, 0, comm)
    if rank == 0
        if noisy
            println(time1, " seconds elapsed in the Lanczos method")
        end

        if Nc == 2
            return 0.5E
        end

        Nbond = length(sys_enl.bonds) + length(env_enl.bonds) + length(superblock_bonds)
        return E + Nbond / Nc
    end

    return 0.0
end

function _report_overlap(Ψ0_guess, Ψ0, comm, rank, noisy)
    if isnothing(Ψ0_guess)
        return nothing
    end

    nume = MPI.Reduce(mydot(Ψ0_guess, Ψ0), MPI.SUM, 0, comm)
    den1 = MPI.Reduce(mydot(Ψ0_guess, Ψ0_guess), MPI.SUM, 0, comm)
    den2 = MPI.Reduce(mydot(Ψ0, Ψ0), MPI.SUM, 0, comm)
    if noisy && rank == 0
        println("overlap |<ψ_guess|ψ>| = ", abs(nume) / sqrt(den1 * den2))
    end
    return nothing
end

function _dmrg_step_tuple(::Val{env_calc}, newblock, newtensor_dict, newblock_enl, truncation_error, energy, Ψ0, trmat, ee, es, Sj) where env_calc
    if env_calc
        return newblock[1], newtensor_dict[1], newblock_enl[1], truncation_error[1], energy, Ψ0, trmat[1], ee[1], es[1], newblock[2], newtensor_dict[2], newblock_enl[2], trmat[2]
    end

    return newblock[1], newtensor_dict[1], newblock_enl[1], truncation_error[1], energy, Ψ0, trmat[1], ee[1], es[1], Sj
end

"""
    dmrg_step!(...)

A single DMRG step. This keeps the historical tuple return value; finite-run
drivers should prefer `dmrg_step_result!` so field access is explicit.
"""
function dmrg_step!(SiSj, sys_label, sys::Block{Nc}, env::Block{Nc}, sys_tensor_dict, env_tensor_dict, sys_enl::EnlargedBlock{Nc}, env_enl::EnlargedBlock{Nc}, Ly, m, α, widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, ::Val{env_calc}; Ψ0_guess = nothing, ES_max = -Inf, correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0), alg = :slow, noisy = true) where {Nc, env_calc}
    workspace = _prepare_step_workspace(sys, env, sys_tensor_dict, env_tensor_dict, sys_enl, env_enl, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, lattice, Val(Nc))
    (; sys_αs, env_αs, sys_βs, env_βs, sys_ms, env_ms, sys_len, env_len, superblock_bonds, bonds_hold, x_conn, y_conn, sys_dp, env_dp, sys_enlarge, env_enlarge, OM, superblock_H1, superblock_H2, sys_connS, env_connS, sys_tensor_dict_hold, env_tensor_dict_hold) = workspace

    Ψ0 = _initial_step_wavefunction(Ψ0_guess, env_ms, sys_ms, OM, Ncpu, rank, engine)
    lanczos_context = _StepLanczosContext(target, comm, rank, engine, alg, superblock_H1, bonds_hold, x_conn, y_conn, sys_connS, env_connS, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, sys_enlarge, env_enlarge, sys_tensor_dict_hold, env_tensor_dict_hold, superblock_H2, OM, sys_len, env_len, Ncpu)
    E, time_Lanczos = _run_step_lanczos!(Ψ0, lanczos_context)
    energy = _step_energy(E, time_Lanczos, sys_enl, env_enl, superblock_bonds, comm, rank, noisy, Val(Nc))
    density_context = _StepDensityContext(comm, rank, Ncpu, engine, α, m, ES_max, noisy)
    measurement_context = _StepMeasurementContext(SiSj, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, comm, rank, Ncpu, engine, correlation, margin, lattice)
    block_context = _StepBlockContext(Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine, lattice)
    correction_context = _StepCorrectionContext(superblock_bonds, sys_connS, env_connS, sys_enl, env_enl, x_conn, y_conn, sys_tensor_dict_hold, env_tensor_dict_hold, sys_tensor_dict, env_tensor_dict, sys_ms, env_ms, sys_dp, env_dp, sys_βs, env_βs, sys_enlarge, env_enlarge, engine)

    newblock, newblock_enl, trmat, newtensor_dict = _step_result_buffers(Val(Nc), engine)

    ee = Float64[]
    es = Dict{SUNIrrep{Nc}, Vector{Float64}}[]
    truncation_error = Float64[]

    for switch in 1 : (env_calc ? 2 : 1)
        side = _step_side_context(switch, sys_label, sys_len, env_len, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, x_conn, y_conn, sys, env, sys_enl, env_enl)
        (; len, ms, dp, conn, label, block, block_enl) = side
        βs = side.betas

        balancer = _density_matrix_balancer(ms, density_context.Ncpu, density_context.rank)
        MPI.Bcast!(balancer, 0, comm)

        dimβ = dim.(βs)
        ρs = _reduced_density_matrices(Ψ0, switch, side, dimβ, sys_len, env_len, density_context)

        if density_context.rank == 0 && density_context.α != 0.0
            _apply_density_matrix_correction!(ρs, switch, side, density_context, correction_context)
        end

        MPI.Barrier(comm)

        ρnew = _balanced_density_matrices(ρs, balancer, side, density_context)

        MPI.Barrier(comm)

        λζtemp, time_DM = _density_eigendecomposition(ρnew, density_context)

        time2 = MPI.Reduce(time_DM, MPI.MAX, 0, comm)

        if noisy && rank == 0 && switch == 1
            println(time2, " seconds elapsed in the density matrix diagonalization")
        end

        MPI.Barrier(comm)

        λ, ζ = _collect_density_eigenpairs(λζtemp, balancer, side, density_context)

        MPI.Barrier(comm)

        ee_value, esi, transformation_matrix, msnew, Hnew, indices = _density_truncation_basis(λ, ζ, side, dimβ, sys_enl, env_enl, density_context, Val(Nc), switch)
        push!(ee, ee_value)
        push!(trmat, transformation_matrix)
        push!(es, esi)

        MPI.Barrier(comm)
        tensor_dict, Sj = _measure_step_tensor_dict!(measurement_context, switch, label, sys, env, sys_enl, env_enl, Ψ0, transformation_matrix, Sj)
        MPI.Barrier(comm)

        _push_step_block!(newblock, newtensor_dict, newblock_enl, tensor_dict, side, msnew, Hnew, block_context, Val(Nc))

        if rank == 0
            push!(truncation_error, _step_truncation_error(λ, indices, dimβ, newblock_enl[switch], Val(Nc)))
            if noisy && switch == 1
                println("truncation error: ", truncation_error[switch])
            end
        else
            push!(truncation_error, 0.0)
        end
    end

    _report_overlap(Ψ0_guess, Ψ0, comm, rank, noisy)
    return _dmrg_step_tuple(Val(env_calc), newblock, newtensor_dict, newblock_enl, truncation_error, energy, Ψ0, trmat, ee, es, Sj)
end

struct DMRGStepResult
    block
    tensor_dict
    block_enl
    trerr
    energy
    Ψ
    trmat
    ee
    es
    Sj
    env_block
    env_tensor_dict
    env_block_enl
    env_trmat
end

function _dmrg_step_result(parts::Tuple)
    if length(parts) ∉ (10, 13)
        throw(ArgumentError("dmrg_step! returned $(length(parts)) values; expected 10 or 13"))
    end

    env_block = length(parts) == 13 ? parts[10] : nothing
    env_tensor_dict = length(parts) == 13 ? parts[11] : nothing
    env_block_enl = length(parts) == 13 ? parts[12] : nothing
    env_trmat = length(parts) == 13 ? parts[13] : nothing

    DMRGStepResult(
        parts[1],
        parts[2],
        parts[3],
        parts[4],
        parts[5],
        parts[6],
        parts[7],
        parts[8],
        parts[9],
        length(parts) == 10 ? parts[10] : nothing,
        env_block,
        env_tensor_dict,
        env_block_enl,
        env_trmat,
    )
end

function dmrg_step_result!(args...; kwargs...)
    _dmrg_step_result(dmrg_step!(args...; kwargs...))
end
