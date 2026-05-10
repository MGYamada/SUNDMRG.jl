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

