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
    empty_engine_matrix_vector(engine)
end

function _syrk!(::Type{<:CPUEngine}, uplo, trans, α, A, β, C)
    BLAS.syrk!(uplo, trans, α, A, β, C)
end

function _syrk!(::Type{<:GPUEngine}, uplo, trans, α, A, β, C)
    CUBLAS.syrk!(uplo, trans, α, A, β, C)
end

function _density_syev!(::Type{<:CPUEngine}, ρ)
    LAPACK.syev!('V', 'U', ρ)
end

function _density_syev!(::Type{<:GPUEngine}, ρ)
    magma_syevd!('V', 'U', ρ)
end

function _density_matrix_from_wavefunction!(ρ, Ψ0, fac, trans, engine)
    for Ψ in Ψ0
        for J in eachindex(Ψ)
            _syrk!(engine, 'U', trans, fac, Ψ[J], 1.0, ρ)
        end
    end
    synchronize_engine(engine)
    return ρ
end

function _reduced_density_matrices(Ψ0, switch, side::_StepSideContext, dimβ, sys_len, env_len, context::_StepDensityContext)
    (; len, ms) = side
    (; comm, rank, engine) = context
    ρs = _empty_density_matrix_vector(engine)

    if switch == 1
        for k in 1 : len
            fac = 1.0 / dimβ[k]
            ρ = zeros_like_engine(engine, Float64, ms[k], ms[k])
            _density_matrix_from_wavefunction!(ρ, (Ψ0[j, k] for j in 1 : env_len), fac, 'T', engine)
            MPI.Reduce!(ρ, MPI.SUM, 0, comm)
            if rank == 0
                push!(ρs, ρ)
            end
        end
    else
        for k in 1 : len
            fac = 1.0 / dimβ[k]
            ρ = zeros_like_engine(engine, Float64, ms[k], ms[k])
            _density_matrix_from_wavefunction!(ρ, (Ψ0[k, j] for j in 1 : sys_len), fac, 'N', engine)
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
    time_DM = @elapsed λζtemp = map(ρ -> _density_syev!(engine, ρ), ρnew)

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
                Stemp = _enlarged_spin_tensor(sys_ms, sys_βs, sys_dp, sys_enlarge, Sx, engine)
            else
                Sx = haskey(env_tensor_dict_hold, x) ? env_tensor_dict_hold[x] : env_tensor_dict[x]
                Stemp = _enlarged_spin_tensor(env_ms, env_βs, env_dp, env_enlarge, Sx, engine)
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
