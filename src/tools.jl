"""
Ψ0_guess = eig_prediction(Ψ0, sys_label, sys_enl, env_enl, sys_trmat, env_trmat, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
Eigenstate prediction
"""
function eig_prediction(Ψ0, sys_label, sys_enl::EnlargedBlock{Nc}, env_enl::EnlargedBlock{Nc}, sys_trmat, env_trmat, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine) where Nc
    γirrep = γ_list[end]

    sys_mαβ = MPI.bcast(sys_enl.mαβ, 0, comm)::Matrix{Int}
    env_mαβ = MPI.bcast(env_enl.mαβ, 0, comm)::Matrix{Int}
    cum_sys_mαβ = vcat(zeros(Int, 1, size(sys_mαβ, 2)), cumsum(sys_mαβ, dims = 1))
    cum_env_mαβ = vcat(zeros(Int, 1, size(env_mαβ, 2)), cumsum(env_mαβ, dims = 1))
    sys_αs = MPI.bcast(sys_enl.α_list, 0, comm)::Vector{SUNIrrep{Nc}}
    sys_βs = MPI.bcast(sys_enl.β_list, 0, comm)::Vector{SUNIrrep{Nc}}
    env_αs = MPI.bcast(env_enl.α_list, 0, comm)::Vector{SUNIrrep{Nc}}
    env_βs = MPI.bcast(env_enl.β_list, 0, comm)::Vector{SUNIrrep{Nc}}
    OM = OM_matrix(sys_βs, env_αs, γirrep)
    sys_mα = MPI.bcast(sys_enl.mα_list, 0, comm)::Vector{Int}
    sys_mβ = MPI.bcast(sys_enl.mβ_list, 0, comm)::Vector{Int}
    env_mα = MPI.bcast(env_enl.mα_list, 0, comm)::Vector{Int}
    env_mβ = MPI.bcast(env_enl.mβ_list, 0, comm)::Vector{Int}

    if engine <: GPUEngine
        Ψ0_guess = [[CUDA.zeros(Float64, mi, mj) for J in 1 : ((j + i - 2) % Ncpu == rank ? OM[j, i] : 0)] for (i, mi) in enumerate(env_mα), (j, mj) in enumerate(sys_mβ)]
    else
        Ψ0_guess = [[zeros(mi, mj) for J in 1 : ((j + i - 2) % Ncpu == rank ? OM[j, i] : 0)] for (i, mi) in enumerate(env_mα), (j, mj) in enumerate(sys_mβ)]
    end
    if rank == 0
        for j in 1 : length(sys_αs)
            MPI.bcast(size(sys_trmat[j]), 0, comm)::Tuple{Int, Int}
            MPI.Bcast!(sys_trmat[j], 0, comm)
        end
        for i in 1 : length(env_αs)
            MPI.bcast(size(env_trmat[i]), 0, comm)::Tuple{Int, Int}
            MPI.Bcast!(env_trmat[i], 0, comm)
        end
    else
        if engine <: GPUEngine
            sys_trmat = CuMatrix{Float64}[]
        else
            sys_trmat = Matrix{Float64}[]
        end
        for j in 1 : length(sys_αs)
            x, y = MPI.bcast(nothing, 0, comm)::Tuple{Int, Int}
            if engine <: GPUEngine
                push!(sys_trmat, CuMatrix{Float64}(undef, x, y))
            else
                push!(sys_trmat, Matrix{Float64}(undef, x, y))
            end
            MPI.Bcast!(sys_trmat[j], 0, comm)
        end
        if engine <: GPUEngine
            env_trmat = CuMatrix{Float64}[]
        else
            env_trmat = Matrix{Float64}[]
        end
        for i in 1 : length(env_αs)
            x, y = MPI.bcast(nothing, 0, comm)::Tuple{Int, Int}
            if engine <: GPUEngine
                push!(env_trmat, CuMatrix{Float64}(undef, x, y))
            else
                push!(env_trmat, Matrix{Float64}(undef, x, y))
            end
            MPI.Bcast!(env_trmat[i], 0, comm)
        end
    end

    OM2 = OM_matrix(env_βs, sys_αs, γirrep)
    for k in 1 : length(env_βs), j in 1 : length(sys_αs)
        for om2 in 1 : OM2[k, j]
            βk = env_βs[k]
            αj = sys_αs[j]
            root = (k + j - 2) % Ncpu
            if rank == root
                temp1 = Ψ0[k, j][om2]
            else
                if engine <: GPUEngine
                    temp1 = CuMatrix{Float64}(undef, env_mβ[k], sys_mα[j])
                else
                    temp1 = Matrix{Float64}(undef, env_mβ[k], sys_mα[j])
                end
            end
            MPI.Bcast!(temp1, root, comm)
            temp2 = temp1 * sys_trmat[j]
            for i in findall(x -> x > 0, env_mαβ[:, k])
                αi = env_αs[i]
                for l in findall(OM[:, i] .> 0)
                    βl = sys_βs[l]
                    if (i + l - 2) % Ncpu == rank && sys_mαβ[j, l] > 0
                        temp3 = env_trmat[i] * temp2[cum_env_mαβ[i, k] + 1 : cum_env_mαβ[i + 1, k], :]
                        cmatrix = on_the_fly ? on_the_fly_calc5(Nc, (αj, βl, αi, γirrep, βk)) : tables[5][αj, βl, αi, γirrep, βk]
                        for om1 in 1 : OM[l, i]
                            @. Ψ0_guess[i, l][om1][:, cum_sys_mαβ[j, l] + 1 : cum_sys_mαβ[j + 1, l]] += cmatrix[om2, om1] * temp3
                        end
                    end
                end
            end
        end
    end

    Ψ0_guess
end

"""
wavefunction_reverse(Ψ0, sys_label, sys_block_enl, env_block_enl, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
exchanges sys and env.
"""
function wavefunction_reverse(Ψ0, sys_label, sys_block_enl::EnlargedBlock{Nc}, env_block_enl::EnlargedBlock{Nc}, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine) where Nc
    _wavefunction_reverse(Ψ0, sys_label, sys_block_enl, env_block_enl, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Nc)
end

"""
wavefunction_reverse(Ψ0, sys_label, sys_block, env_block, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
exchanges sys and env.
"""
function wavefunction_reverse(Ψ0, sys_label, sys_block::Block{Nc}, env_block::Block{Nc}, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine) where Nc
    _wavefunction_reverse(Ψ0, sys_label, sys_block, env_block, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Nc)
end

function _wavefunction_reverse(Ψ0, sys_label, sys, env, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Nc)
    γirrep = γ_list[mod1(sys.length + env.length, Nc)]

    sys_βs = MPI.bcast(sys.β_list, 0, comm)::Vector{SUNIrrep{Nc}}
    env_βs = MPI.bcast(env.β_list, 0, comm)::Vector{SUNIrrep{Nc}}

    if engine <: GPUEngine
        Ψ0_new = [[CUDA.zeros(Float64, size(Ψ0[i, j][J], 2), size(Ψ0[i, j][J], 1)) for J in 1 : length(Ψ0[i, j])] for j in 1 : size(Ψ0, 2), i in 1 : size(Ψ0, 1)]
    else
        Ψ0_new = [[zeros(size(Ψ0[i, j][J], 2), size(Ψ0[i, j][J], 1)) for J in 1 : length(Ψ0[i, j])] for j in 1 : size(Ψ0, 2), i in 1 : size(Ψ0, 1)]
    end

    for j in 1 : size(Ψ0, 2), i in 1 : size(Ψ0, 1)
        OM = length(Ψ0[i, j])
        if OM > 0
            rmatrix = (on_the_fly ? on_the_fly_calc6((sys_βs[j], env_βs[i], γirrep)) : tables[6][sys_βs[j], env_βs[i], γirrep])
            for τ in 1 : OM, τ′ in 1 : OM
                @. Ψ0_new[j, i][τ′] += rmatrix[τ′, τ] * Ψ0[i, j][τ]'
            end
        end
    end

    Ψ0_new
end

"""
value = on_the_fly_calc1(Nc, key)
On-the-fly calculation of tables[1].
"""
function on_the_fly_calc1(Nc, key)
    trivial = trivialirrep(Val(Nc))
    funda = fundamentalirrep(Val(Nc))
    adjoint = adjointirrep(Val(Nc))

    α1, β1, α2, β2 = key
    wigner9ν(α1, funda, β1, adjoint, trivial, adjoint, α2, funda, β2)[:, 1, 1, 1, 1, :]
end

"""
value = on_the_fly_calc2(Nc, key)
On-the-fly calculation of tables[2].
"""
function on_the_fly_calc2(Nc, key)
    funda = fundamentalirrep(Val(Nc))
    adjoint = adjointirrep(Val(Nc))

    α, β1, β2 = key
    wigner6ν(α, funda, β1, adjoint, β2, funda)[1, 1, 1, :]
end

"""
value = on_the_fly_calc3(Nc, key)
On-the-fly calculation of tables[3].
"""
function on_the_fly_calc3(Nc, key)
    trivial = trivialirrep(Val(Nc))
    funda = fundamentalirrep(Val(Nc))
    adjoint = adjointirrep(Val(Nc))

    α1, β, α2 = key
    wigner9ν(α1, funda, β, adjoint, adjoint, trivial, α2, funda, β)[:, 1, 1, 1, 1, 1]
end

"""
value = on_the_fly_calc4(Nc, key)
On-the-fly calculation of tables[4].
"""
function on_the_fly_calc4(Nc, key)
    trivial = trivialirrep(Val(Nc))
    adjoint = adjointirrep(Val(Nc))

    α1, β1, γ, α2, β2 = key
    wigner9ν(α1, β1, γ, adjoint, adjoint, trivial, α2, β2, γ)[:, :, :, :, 1, 1]
end

"""
value = on_the_fly_calc5(Nc, key)
On-the-fly calculation of tables[5].
"""
function on_the_fly_calc5(Nc, key)
    funda = fundamentalirrep(Val(Nc))

    αj, βl, αi, γ, βk = key
    wigner6ν(αj, funda, βl, αi, γ, βk)[1, :, 1, :] .* wigner3ν(αi, funda, βk)[1, 1]
end

"""
value = on_the_fly_calc6(key)
On-the-fly calculation of tables[6].
"""
function on_the_fly_calc6(key)
    wigner3ν(key...)
end