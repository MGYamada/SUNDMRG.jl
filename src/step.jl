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
    if engine <: GPUEngine
        Ψtemp = [[CUDA.zeros(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
    else
        Ψtemp = [[zeros(env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
    end

    for s in superblock_H2
        root1 = (s.sys_in + s.env_in - 2) % Ncpu
        if rank == root1
            temp3 = Ψin[s.env_in, s.sys_in][s.om1]
        else
            if engine <: GPUEngine
                temp3 = CuMatrix{Float64}(undef, s.env_in_size, s.sys_in_size)
            else
                temp3 = Matrix{Float64}(undef, s.env_in_size, s.sys_in_size)
            end
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

"""
newblock, newtensor_dict, newblock_enl, trerr, energy, Ψ0, trmat, ee, es, Sj = dmrg_step!(SiSj, sys_label, sys, env, sys_tensor_dict, env_tensor_dict, sys_enl, env_enl, Ly, m, α, widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(env_calc); Ψ0_guess = nothing, ES_max = -Inf, correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0), alg = :slow, noisy = true)
a single step for DMRG
"""
function dmrg_step!(SiSj, sys_label, sys::Block{Nc}, env::Block{Nc}, sys_tensor_dict, env_tensor_dict, sys_enl::EnlargedBlock{Nc}, env_enl::EnlargedBlock{Nc}, Ly, m, α, widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, ::Val{env_calc}; Ψ0_guess = nothing, ES_max = -Inf, correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0), alg = :slow, noisy = true) where {Nc, env_calc}
    sys_αs = MPI.bcast(sys_enl.α_list, 0, comm)::Vector{SUNIrrep{Nc}}
    env_αs = MPI.bcast(env_enl.α_list, 0, comm)::Vector{SUNIrrep{Nc}}
    sys_βs = MPI.bcast(sys_enl.β_list, 0, comm)::Vector{SUNIrrep{Nc}}
    env_βs = MPI.bcast(env_enl.β_list, 0, comm)::Vector{SUNIrrep{Nc}}
    sys_ms = MPI.bcast(sys_enl.mβ_list, 0, comm)::Vector{Int}
    env_ms = MPI.bcast(env_enl.mβ_list, 0, comm)::Vector{Int}
    sys_mαβ = MPI.bcast(sys_enl.mαβ, 0, comm)::Matrix{Int}
    env_mαβ = MPI.bcast(env_enl.mαβ, 0, comm)::Matrix{Int}
    sys_αβmatrix = sys_mαβ .> 0
    env_αβmatrix = env_mαβ .> 0
    sys_len = length(sys_βs)
    env_len = length(env_βs)
    cum_sys_mαβ = vcat(zeros(Int, 1, sys_len), cumsum(sys_mαβ, dims = 1))
    cum_env_mαβ = vcat(zeros(Int, 1, env_len), cumsum(env_mαβ, dims = 1))

    γirrep = γ_list[mod1(sys_enl.length + env_enl.length, Nc)]
    adjoint = adjointirrep(Val(Nc))
    trivial = trivialirrep(Val(Nc))

    superblock_bonds = [(y, y) for y in 1 : min(sys_enl.length, env_enl.length, Ly)]
    x_conn = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(sys_enl.length, 2Ly))
    y_conn = (x -> x <= Ly ? x : 2Ly + 1 - x)(mod1(env_enl.length, 2Ly))
    if lattice == :honeycombZC
        filter!(z -> z[1] <= min(x_conn, y_conn) ? iseven(z[1]) : isodd(z[1]), superblock_bonds)
    end
    if (x_conn, y_conn) ∉ superblock_bonds && !(lattice == :honeycombZC && sys_enl.length <= Ly && env_enl.length <= Ly)
        push!(superblock_bonds, (x_conn, y_conn))
    end

    sort!(superblock_bonds)
    if (sys_enl.length >= Ly || env_enl.length >= Ly) && sys_enl.length % Ly != 0
        if mod1(sys_enl.length, 2Ly) <= Ly
            push!(superblock_bonds, (1, Ly))
        else
            push!(superblock_bonds, (Ly, 1))
        end
    end
    dc = dcinit(length(superblock_bonds), Ncpu)
    bonds_hold = superblock_bonds[dc[rank + 1] + 1 : dc[rank + 2]]

    if engine <: GPUEngine
        superblock_H1 = SuperBlock1GPU[]
    else
        superblock_H1 = SuperBlock1CPU[]
    end
    superblock_H2 = SuperBlock2[]

    sys_dp = map(x -> directproduct(x, adjoint), sys_βs)
    env_dp = map(x -> directproduct(x, adjoint), env_βs)

    sys_dp2 = map(x -> directproduct(x, adjoint), sys_αs)
    sys_enlarge = BlockEnlarging[]
    for i in 1 : sys_len, j in 1 : sys_len
        if haskey(sys_dp[j], sys_βs[i])
            for ki in findall(sys_αβmatrix[:, i]), kj in findall(sys_αβmatrix[:, j])
                if haskey(sys_dp2[kj], sys_αs[ki])
                    amatrix = on_the_fly ? on_the_fly_calc1(Nc, (sys_αs[kj], sys_βs[j], sys_αs[ki], sys_βs[i])) : tables[1][sys_αs[kj], sys_βs[j], sys_αs[ki], sys_βs[i]]
                    for τ1 in 1 : size(amatrix, 2), τ2 in 1 : size(amatrix, 1)
                        push!(sys_enlarge, BlockEnlarging(amatrix[τ2, τ1], τ1, τ2, cum_sys_mαβ[ki, i] + 1 : cum_sys_mαβ[ki + 1, i], cum_sys_mαβ[kj, j] + 1 : cum_sys_mαβ[kj + 1, j], i, j, ki, kj))
                    end
                end
            end
        end
    end

    env_dp2 = map(x -> directproduct(x, adjoint), env_αs)
    env_enlarge = BlockEnlarging[]
    for i in 1 : env_len, j in 1 : env_len
        if haskey(env_dp[j], env_βs[i])
            for ki in findall(env_αβmatrix[:, i]), kj in findall(env_αβmatrix[:, j])
                if haskey(env_dp2[kj], env_αs[ki])
                    bmatrix = on_the_fly ? on_the_fly_calc1(Nc, (env_αs[kj], env_βs[j], env_αs[ki], env_βs[i])) : tables[1][env_αs[kj], env_βs[j], env_αs[ki], env_βs[i]]
                    for τ1 in 1 : size(bmatrix, 2), τ2 in 1 : size(bmatrix, 1)
                        push!(env_enlarge, BlockEnlarging(bmatrix[τ2, τ1], τ1, τ2, cum_env_mαβ[ki, i] + 1 : cum_env_mαβ[ki + 1, i], cum_env_mαβ[kj, j] + 1 : cum_env_mαβ[kj + 1, j], i, j, ki, kj))
                    end
                end
            end
        end
    end

    OM = OM_matrix(sys_βs, env_βs, γirrep)
    for k1 in 1 : sys_len, k2 in 1 : env_len
        if OM[k1, k2] > 0 && (k1 + k2 - 2) % Ncpu == rank
            if engine <: GPUEngine
                push!(superblock_H1, SuperBlock1GPU(sys_enl.scalar_dict[:H][k1], env_enl.scalar_dict[:H][k2], k1, k2))
            else
                push!(superblock_H1, SuperBlock1CPU(sys_enl.scalar_dict[:H][k1], env_enl.scalar_dict[:H][k2], k1, k2))
            end
        end
    end

    fac = sqrt(Nc ^ 2 - 1) * signfactor
    for k1 in 1 : sys_len, k2 in 1 : env_len
        for om1 in 1 : OM[k1, k2]
            miniblock = MiniBlock[]
            for k3 in 1 : sys_len
                if haskey(sys_dp[k1], sys_βs[k3])
                    for k4 in 1 : env_len
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

    if x_conn ∈ first.(bonds_hold)
        # if engine <: GPUEngine
        #     sys_connS = [@. CUSPARSE.CuSparseMatrixCSC(sys_enl.tensor_dict[x_conn][i, j]) for i in 1 : sys_len, j in 1 : sys_len]
        # else
        sys_connS = sys_enl.tensor_dict[x_conn]
        # end
    else
        sys_connS = nothing
    end
    if y_conn ∈ last.(bonds_hold)
        # if engine <: GPUEngine
        #     env_connS = [@. CUSPARSE.CuSparseMatrixCSC(env_enl.tensor_dict[y_conn][i, j]) for i in 1 : env_len, j in 1 : env_len]
        # else
        env_connS = env_enl.tensor_dict[y_conn]
        # end
    else
        env_connS = nothing
    end

    if engine <: GPUEngine
        sys_tensor_dict_hold = Dict{Int, Matrix{Vector{CuMatrix{Float64}}}}()
    else
        sys_tensor_dict_hold = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
    end
    len = length(sys_αs)

    if rank == 0
        x_list = MPI.bcast(collect(keys(sys_tensor_dict)), 0, comm)::Vector{Int}
    else
        x_list = MPI.bcast(nothing, 0, comm)::Vector{Int}
    end

    for x in x_list
        if x != x_conn
            if rank == 0
                Sold = sys_tensor_dict[x]
                mms = MPI.bcast(sys.mβ_list, 0, comm)::Vector{Int}
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

            if x ∈ first.(bonds_hold)
                if engine <: GPUEngine
                    sys_tensor_dict_hold[x] = [CuArray.(Sold[i, j]) for i in 1 : len, j in 1 : len]
                else
                    sys_tensor_dict_hold[x] = Sold
                end
            end
        end
    end

    if engine <: GPUEngine
        env_tensor_dict_hold = Dict{Int, Matrix{Vector{CuMatrix{Float64}}}}()
    else
        env_tensor_dict_hold = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
    end
    len = length(env_αs)

    if rank == 0
        y_list = MPI.bcast(collect(keys(env_tensor_dict)), 0, comm)::Vector{Int}
    else
        y_list = MPI.bcast(nothing, 0, comm)::Vector{Int}
    end

    for y in y_list
        if y != y_conn
            if rank == 0
                Sold = env_tensor_dict[y]
                mms = MPI.bcast(env.mβ_list, 0, comm)::Vector{Int}
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

            if y ∈ last.(bonds_hold)
                # if engine <: GPUEngine
                #     env_tensor_dict_hold[y] =  [CuArray.(Sold[i, j]) for i in 1 : len, j in 1 : len]
                # else
                env_tensor_dict_hold[y] =  Sold
                # end
            end
        end
    end

    holdmax = maximum(dc[2 : end] .- dc[1 : end - 1])
    while length(bonds_hold) < holdmax
        push!(bonds_hold, (0, 0))
    end

    if isnothing(Ψ0_guess)
        if engine <: GPUEngine
            Ψ0 = [[CUDA.rand(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : ((kj + ki - 2) % Ncpu == rank ? OM[kj, ki] : 0)] for ki in 1 : env_len, kj in 1 : sys_len]
        else
            Ψ0 = [[rand(env_ms[ki], sys_ms[kj]) for J in 1 : ((kj + ki - 2) % Ncpu == rank ? OM[kj, ki] : 0)] for ki in 1 : env_len, kj in 1 : sys_len]
        end
    else
        Ψ0 = deepcopy(Ψ0_guess)
    end

    time_Lanczos = @elapsed E = Lanczos!(Ψ0, target + 1, comm, rank, engine; alg = alg) do Ψout, Ψin
        for s in superblock_H1
            for J in eachindex(Ψin[s.env_ind, s.sys_ind])
                temp1 = Ψin[s.env_ind, s.sys_ind][J] * s.sys_H'
                temp2 = s.env_H * Ψin[s.env_ind, s.sys_ind][J]
                @. Ψout[s.env_ind, s.sys_ind][J] = temp1 + temp2
            end
        end

        for (x, y) in bonds_hold
            if x == x_conn
                sys_S = sys_connS
            elseif x > 0
                if engine <: GPUEngine
                    sys_S = [[CUDA.zeros(Float64, sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                else
                    sys_S = [[zeros(sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                end
                for e in sys_enlarge
                    @. sys_S[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * sys_tensor_dict_hold[x][e.ki, e.kj][e.τ2]
                end
            else
                sys_S = nothing
            end

            if y == y_conn
                env_S = env_connS
            elseif y > 0
                if engine <: GPUEngine
                    env_S = [[CUDA.zeros(Float64, env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                else
                    env_S = [[zeros(env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                end
                for e in env_enlarge
                    @. env_S[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * env_tensor_dict_hold[y][e.ki, e.kj][e.τ2]
                end
            else
                env_S = nothing
            end

            Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
        end
    end

    time1 = MPI.Reduce(time_Lanczos, MPI.MAX, 0, comm)

    if rank == 0
        if noisy
            println(time1, " seconds elapsed in the Lanczos method")
        end

        if Nc == 2
            energy = 0.5E
        else
            Nbond = length(sys_enl.bonds) + length(env_enl.bonds) + length(superblock_bonds)
            energy = E + Nbond / Nc
        end
    else
        energy = 0.0
    end

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

    ee = Float64[]
    es = Dict{SUNIrrep{Nc}, Vector{Float64}}[]
    truncation_error = Float64[]

    env_label = sys_label == :l ? :r : :l

    for switch in 1 : (env_calc ? 2 : 1)
        len = [sys_len, env_len][switch]
        ms = [sys_ms, env_ms][switch]
        βs = [sys_βs, env_βs][switch]
        dp = [sys_dp, env_dp][switch]
        conn = [x_conn, y_conn][switch]
        label = [sys_label, env_label][switch]
        block = [sys, env][switch]
        block_enl = [sys_enl, env_enl][switch]

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
        MPI.Bcast!(balancer, 0, comm)

        dimβ = dim.(βs)
        if rank == 0
            if engine <: GPUEngine
                ρs = CuMatrix{Float64}[]
            else
                ρs = Matrix{Float64}[]
            end
        end

        if switch == 1
            for k in 1 : len
                fac = 1.0 / dimβ[k]
                if engine <: GPUEngine
                    ρ = CUDA.zeros(Float64, ms[k], ms[k])
                    for j in 1 : env_len
                        for J in eachindex(Ψ0[j, k])
                            CUBLAS.syrk!('U', 'T', fac, Ψ0[j, k][J], 1.0, ρ)
                        end
                    end
                    CUDA.synchronize()
                else
                    ρ = zeros(ms[k], ms[k])
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
                    ρ = CUDA.zeros(Float64, ms[k], ms[k])
                    for j in 1 : sys_len
                        for J in eachindex(Ψ0[k, j])
                            CUBLAS.syrk!('U', 'N', fac, Ψ0[k, j][J], 1.0, ρ)
                        end
                    end
                    CUDA.synchronize()
                else
                    ρ = zeros(ms[k], ms[k])
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

        if rank == 0 && α != 0.0
            if engine <: GPUEngine
                Sρs = [CUDA.zeros(Float64, ms[k], ms[k]) for k in 1 : len]
            else
                Sρs = [zeros(ms[k], ms[k]) for k in 1 : len]
            end
            for x in unique(map(x -> x[switch], superblock_bonds))
                if x == conn
                    if switch == 1
                        if isnothing(sys_connS)
                            # if engine <: GPUEngine
                            #     Stemp = [@. CUSPARSE.CuSparseMatrixCSC(sys_enl.tensor_dict[x_conn][i, j]) for i in 1 : sys_len, j in 1 : sys_len]
                            # else
                            Stemp = sys_enl.tensor_dict[x_conn]
                            # end
                        else
                            Stemp = sys_connS
                        end
                    else
                        if isnothing(env_connS)
                            # if engine <: GPUEngine
                            #     Stemp = [@. CUSPARSE.CuSparseMatrixCSC(env_enl.tensor_dict[y_conn][i, j]) for i in 1 : env_len, j in 1 : env_len]
                            # else
                            Stemp = env_enl.tensor_dict[y_conn]
                            # end
                        else
                            Stemp = env_connS
                        end
                    end
                else
                    if switch == 1
                        if haskey(sys_tensor_dict_hold, x)
                            Sx = sys_tensor_dict_hold[x]
                        else
                            temp_len = length(sys_αs)
                            # if engine <: GPUEngine
                            #     Sx = [CuArray.(sys_tensor_dict[x][i, j]) for i in 1 : temp_len, j in 1 : temp_len]
                            # else
                            Sx = sys_tensor_dict[x]
                            # end
                        end
                        if engine <: GPUEngine
                            Stemp = [[CUDA.zeros(Float64, sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                        else
                            Stemp = [[zeros(sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                        end
                        for e in sys_enlarge
                            @. Stemp[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sx[e.ki, e.kj][e.τ2]
                        end
                    else
                        if haskey(env_tensor_dict_hold, x)
                            Sx = env_tensor_dict_hold[x]
                        else
                            temp_len = length(env_αs)
                            # if engine <: GPUEngine
                            #     Sx = [CuArray.(env_tensor_dict[x][i, j]) for i in 1 : temp_len, j in 1 : temp_len]
                            # else
                            Sx = env_tensor_dict[x]
                            # end
                        end
                        if engine <: GPUEngine
                            Stemp = [[CUDA.zeros(Float64, env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                        else
                            Stemp = [[zeros(env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                        end
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
        end

        MPI.Barrier(comm)

        if rank == 0
            ρnew = ρs[balancer .== 0]
        else
            if engine <: GPUEngine
                ρnew = CuMatrix{Float64}[]
            else
                ρnew = Matrix{Float64}[]
            end
        end
        for k in 1 : len
            if balancer[k] != 0
                if rank == 0
                    MPI.Send(ρs[k], balancer[k], k, comm)
                elseif rank == balancer[k]
                    if engine <: GPUEngine
                        ρ = CuMatrix{Float64}(undef, ms[k], ms[k])
                    else
                        ρ = Matrix{Float64}(undef, ms[k], ms[k])
                    end
                    MPI.Recv!(ρ, 0, k, comm)
                    push!(ρnew, ρ)
                end
            end
        end

        MPI.Barrier(comm)

        time_DM = @elapsed if engine <: GPUEngine
            λζtemp = map(ρ -> magma_syevd!('V', 'U', ρ), ρnew)
        else
            λζtemp = map(ρ -> LAPACK.syev!('V', 'U', ρ), ρnew)
        end

        time2 = MPI.Reduce(time_DM, MPI.MAX, 0, comm)

        if noisy && rank == 0 && switch == 1
            println(time2, " seconds elapsed in the density matrix diagonalization")
        end

        MPI.Barrier(comm)

        if rank == 0
            λ = [Vector{Float64}(undef, ms[k]) for k in 1 : len]
            ζ = [Matrix{Float64}(undef, ms[k], ms[k]) for k in 1 : len]
            @. λ[balancer .== 0] = first(λζtemp)
            @. ζ[balancer .== 0] = last(λζtemp)
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

        MPI.Barrier(comm)

        esi = Dict{SUNIrrep{Nc}, Vector{Float64}}()
        if rank == 0
            push!(ee, -sum(@. dimβ * sum(map(x -> x * log(x), filter(x -> x > 0.0, λ)))))
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
            if engine <: GPUEngine
                transformation_matrix = map(x -> CuArray(x[1][:, x[2]]), zip(ζ, indices))
            else
                transformation_matrix = map(x -> x[1][:, x[2]], zip(ζ, indices))
            end

            msnew = map(x -> size(x, 2), transformation_matrix)
            if noisy && switch == 1
                println("Keeping ", sum(msnew), " SU($Nc) states equivalent to ", sum(dimβ .* msnew), " U(1) states")
            end
            Hnew = map(x -> Array(x[2]' * (x[1] * x[2])), zip(block_enl.scalar_dict[:H], transformation_matrix))
        else
            push!(ee, 0.0)
            if engine <: GPUEngine
                transformation_matrix = [CuMatrix{Float64}(undef, 0, 0)]
            else
                transformation_matrix = [Matrix{Float64}(undef, 0, 0)]
            end
        end
        push!(trmat, transformation_matrix)
        push!(es, esi)

        MPI.Barrier(comm)
        if switch == 1
            tensor_dict, Sj = measurement!(SiSj, label, sys, env, sys_enl, env_enl, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = correlation, margin = margin, lattice = lattice, Sj = Sj)
        else
            # Strictly speaking, superblock_H2 is not exchanged here, but don't worry. It works.
            tensor_dict, Sj = measurement!(SiSj, label, env, sys, env_enl, sys_enl, Ly, y_conn, x_conn, env_connS, sys_connS, env_len, sys_len, env_tensor_dict, sys_tensor_dict, env_tensor_dict_hold, sys_tensor_dict_hold, env_αs, sys_αs, env_βs, sys_βs, env_dp, sys_dp, env_ms, sys_ms, env_enlarge, sys_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = correlation, margin = margin, lattice = lattice, Sj = Sj)
        end
        MPI.Barrier(comm)

        if rank == 0
            push!(newblock, Block(block_enl.length, block_enl.bonds, block_enl.β_list, block_enl.mβ_list, msnew, Dict{Symbol, Vector{Matrix{Float64}}}(:H => Hnew)))
        else
            push!(newblock, Block(block_enl.length, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}()))
        end

        push!(newtensor_dict, tensor_dict)
        push!(newblock_enl, enlarge_block(newblock[switch], tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice))

        if rank == 0
            λnew = map(x -> x[1][x[2]], zip(λ, indices))
            push!(truncation_error, sum(sort(@. sum(λ) * dimβ; by = abs)) - sum(map(x -> sum(sort(sum.(λnew[newblock_enl[switch].mαβ[:, x[1]] .> 0]); by = abs)) * dim(x[2]), enumerate(newblock_enl[switch].β_list))) / Nc)
            if noisy && switch == 1
                println("truncation error: ", truncation_error[switch])
            end
        else
            push!(truncation_error, 0.0)
        end
    end

    if !isnothing(Ψ0_guess)
        nume = MPI.Reduce(dot(Ψ0_guess, Ψ0), MPI.SUM, 0, comm)
        den1 = MPI.Reduce(dot(Ψ0_guess, Ψ0_guess), MPI.SUM, 0, comm)
        den2 = MPI.Reduce(dot(Ψ0, Ψ0), MPI.SUM, 0, comm)
        if noisy && rank == 0
            println("overlap |<ψ_guess|ψ>| = ", abs(nume) / sqrt(den1 * den2))
        end
    end

    if env_calc
        newblock[1], newtensor_dict[1], newblock_enl[1], truncation_error[1], energy, Ψ0, trmat[1], ee[1], es[1], newblock[2], newtensor_dict[2], newblock_enl[2], trmat[2]
    else
        newblock[1], newtensor_dict[1], newblock_enl[1], truncation_error[1], energy, Ψ0, trmat[1], ee[1], es[1], Sj
    end
end