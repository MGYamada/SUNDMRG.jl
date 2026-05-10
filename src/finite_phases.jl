function _record_phase_result!(m_list, errors, energies, EEs, m, result::DMRGStepResult)
    push!(m_list, m)
    push!(errors, result.trerr)
    push!(energies, result.energy)
    push!(EEs, result.ee)
    return result.es
end

function _log_phase_result(runtime::_FiniteRuntime, verbose, energy, L, ee)
    if verbose
        root_println(runtime, "E / N = ", energy / L)
        root_println(runtime, "E     = ", energy)
        root_println(runtime, "S_EE  = ", ee)
    end
    return nothing
end

function _save_edge_blocks!(storage, mirror, left_block, left_trmat, right_block, right_trmat, runtime::_FiniteRuntime)
    if !isroot(runtime)
        return nothing
    end

    save_block_and_trmat!(storage, :l, left_block, left_trmat)
    if mirror
        save_block_and_trmat!(storage, :r, left_block, left_trmat)
    else
        save_block_and_trmat!(storage, :r, right_block, right_trmat)
    end
    return nothing
end

function _init_state(config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; lattice, Lx, Ly, Nc, target, m_warmup, widthmax, tables, fileio, scratch, verbose) = config
    (; engine, on_the_fly, mirror, γ_type, comm, rank, Ncpu, signfactor) = runtime

    m_list = Tuple{Int, Float64}[]
    errors = Float64[]
    energies = Float64[]
    EEs = Float64[]
    EE = Vector{Float64}(undef, Lx - 1)
    ES = Dict{SUNIrrep{Nc}, Vector{Float64}}()
    SiSj = Dict{Tuple{Int, Int}, Float64}()

    block_table = Dict{Tuple{Symbol, Int}, Block{Nc}}()
    tensor_table = Dict{Tuple{Symbol, Int, Int}, Matrix{Vector{Matrix{Float64}}}}()
    trmat_table = Dict{Tuple{Symbol, Int}, Vector{Matrix{Float64}}}()

    if isroot(runtime)
        if verbose
            root_println(runtime, repeat("-", 60))
            root_println(runtime, "SU($Nc) DMRG simulation on the $Lx x $Ly $lattice lattice cylinder:")
            if on_the_fly
                root_println(runtime, "All representations are used in the calculation.")
            else
                irreps = irreplist(Nc, widthmax)
                root_println(runtime, length(irreps), " irreps from ", weight(first(irreps)), " to ", weight(last(irreps)), " are used in the calculation.")
            end
            root_println(runtime, target == 0 ? "The ground state" : "The excited state #$target", " will be calculated.")
            root_println(runtime, repeat("-", 60))
        end

        storage = init_internal_storage(fileio, scratch, block_table, trmat_table, tensor_table, rank)

        blockL = Block(0, Tuple{Int, Int}[], [trivialirrep(Val(Nc))], [1], [1], Dict{Symbol, Vector{Matrix{Float64}}}(:H => [zeros(1, 1)]))
        blockL_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
        trmatL = [to_engine_array(engine, diagm([1.0]))]

        if !mirror
            blockR = Block(0, Tuple{Int, Int}[], [trivialirrep(Val(Nc))], [1], [1], Dict{Symbol, Vector{Matrix{Float64}}}(:H => [zeros(1, 1)]))
            blockR_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
            trmatR = [to_engine_array(engine, diagm([1.0]))]
        else
            blockR = blockL
            blockR_tensor_dict = blockL_tensor_dict
            trmatR = trmatL
        end

        _save_edge_blocks!(storage, mirror, blockL, trmatL, blockR, trmatR, runtime)

        if verbose
            root_println(runtime, "#")
            root_println(runtime, "# Warming up with (m, α) = ", m_warmup)
            root_println(runtime, "#")
        end
    else
        storage = init_internal_storage(fileio, scratch, block_table, trmat_table, tensor_table, rank)
        blockL = Block(0, Tuple{Int, Int}[], γ_type[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
        blockL_tensor_dict = empty_engine_tensor_dict(engine)
        trmatL = empty_engine_matrix_vector(engine)

        if !mirror
            blockR = Block(0, Tuple{Int, Int}[], γ_type[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
            blockR_tensor_dict = empty_engine_tensor_dict(engine)
            trmatR = empty_engine_matrix_vector(engine)
        else
            blockR = blockL
            blockR_tensor_dict = blockL_tensor_dict
            trmatR = empty_engine_matrix_vector(engine)
        end
    end

    blockL_enl = enlarge_block(blockL, blockL_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    if !mirror
        blockR_enl = enlarge_block(blockR, blockR_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    else
        blockR_enl = blockL_enl
    end

    Ψ = empty_engine_tensor_matrices(engine, mirror ? 1 : 2)

    return m_list, errors, energies, EEs, EE, ES, SiSj, storage, blockL, blockL_tensor_dict, blockR, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ
end

function _warmup_phase!(SiSj, blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ, m_list, errors, energies, EEs, ES, storage, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; lattice, Ly, N, m_warmup, widthmax, target, tables, alg, verbose) = config
    (; engine, comm, rank, Ncpu, on_the_fly, mirror, γ_list, signfactor) = runtime

    while blockL.length < Ly
        if verbose && isroot(runtime)
            if mirror
                root_println(runtime, graphic(blockL, blockL))
            else
                root_println(runtime, graphic(blockL, blockR))
            end
        end

        if mirror
            result = dmrg_step_result!(SiSj, :l, blockL, blockL, blockL_tensor_dict, blockL_tensor_dict, blockL_enl, blockL_enl, Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); lattice = lattice, alg = alg, noisy = verbose)
            blockL, blockL_tensor_dict, blockL_enl, trerr, energy, Ψ[1], trmatL, ee, es = result.block, result.tensor_dict, result.block_enl, result.trerr, result.energy, result.Ψ, result.trmat, result.ee, result.es
        else
            result = dmrg_step_result!(SiSj, :l, blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(true); lattice = lattice, alg = alg, noisy = verbose)
            blockL, blockL_tensor_dict, blockL_enl, trerr, energy, Ψ[1], trmatL, ee, es = result.block, result.tensor_dict, result.block_enl, result.trerr, result.energy, result.Ψ, result.trmat, result.ee, result.es
            blockR, blockR_tensor_dict, blockR_enl, trmatR = result.env_block, result.env_tensor_dict, result.env_block_enl, result.env_trmat
            Ψ[2] = wavefunction_reverse(Ψ[1], :l, blockL, blockR, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
        end

        if 2blockL.length == N
            ES = _record_phase_result!(m_list, errors, energies, EEs, m_warmup, result)
        end

        _log_phase_result(runtime, verbose, energy, 2blockL.length, ee)
        _save_edge_blocks!(storage, mirror, blockL, trmatL, blockR, trmatR, runtime)
    end

    return blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ, ES
end

function _growth_phase!(SiSj, blockL::Block{Nc}, blockR::Block{Nc}, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ, m_list, errors, energies, EEs, ES, storage, config::_FiniteRunConfig, runtime::_FiniteRuntime) where Nc
    (; lattice, Ly, N, m_warmup, widthmax, target, tables, alg, verbose) = config
    (; engine, comm, rank, Ncpu, on_the_fly, mirror, γ_type, γ_list, signfactor) = runtime

    L = 2blockL.length

    if mirror
        sys_blocks = [blockL]
        sys_tensor_dicts = [blockL_tensor_dict]
        sys_trmats = [trmatL]
        sys_block_enls = [blockL_enl]
        env_trmats = [trmatL]
        env_block_enls = [blockL_enl]
    else
        sys_blocks = [blockL, blockR]
        sys_tensor_dicts = [blockL_tensor_dict, blockR_tensor_dict]
        sys_trmats = [trmatL, trmatR]
        sys_block_enls = [blockL_enl, blockR_enl]
        env_trmats = [trmatR, trmatL]
        env_block_enls = [blockR_enl, blockL_enl]
    end

    while L < N
        if sys_block_enls[1].length % Ly == 0
            if verbose && isroot(runtime)
                if mirror
                    root_println(runtime, graphic(sys_blocks[1], sys_blocks[1]; sys_label = :l))
                else
                    root_println(runtime, graphic(sys_blocks[1], sys_blocks[2]; sys_label = :l))
                end
            end

            L = 2sys_block_enls[1].length
            if mirror
                result = dmrg_step_result!(SiSj, :l, sys_blocks[1], sys_blocks[1], sys_tensor_dicts[1], sys_tensor_dicts[1], sys_block_enls[1], sys_block_enls[1], Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); lattice = lattice, alg = alg, noisy = verbose)
                sys_blocks[1], sys_tensor_dicts[1], sys_block_enls[1], trerr, energy, Ψ[1], sys_trmats[1], ee, es = result.block, result.tensor_dict, result.block_enl, result.trerr, result.energy, result.Ψ, result.trmat, result.ee, result.es
            else
                result = dmrg_step_result!(SiSj, :l, sys_blocks[1], sys_blocks[2], sys_tensor_dicts[1], sys_tensor_dicts[2], sys_block_enls[1], sys_block_enls[2], Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(true); lattice = lattice, alg = alg, noisy = verbose)
                sys_blocks[1], sys_tensor_dicts[1], sys_block_enls[1], trerr, energy, Ψ[1], sys_trmats[1], ee, es = result.block, result.tensor_dict, result.block_enl, result.trerr, result.energy, result.Ψ, result.trmat, result.ee, result.es
                sys_blocks[2], sys_tensor_dicts[2], sys_block_enls[2], sys_trmats[2] = result.env_block, result.env_tensor_dict, result.env_block_enl, result.env_trmat
                Ψ[2] = wavefunction_reverse(Ψ[1], :l, sys_blocks[1], sys_blocks[2], widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
            end

            _log_phase_result(runtime, verbose, energy, L, ee)
        else
            for i in 1 : (mirror ? 1 : 2)
                if i == 1
                    sys_label, env_label = :l, :r
                else
                    sys_label, env_label = :r, :l
                end
                if sys_blocks[i].length % Ly == 0
                    if isroot(runtime)
                        env_block, env_trmats[i] = load_block_and_trmat(storage, env_label, L - sys_blocks[i].length - 1, engine, Val(Nc))

                        env_tensor_dict = spin_operators!(storage, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
                    else
                        env_block = Block(L - sys_blocks[i].length - 1, Tuple{Int, Int}[], γ_type[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
                        env_tensor_dict = empty_engine_tensor_dict(engine)
                        env_trmats[i] = empty_engine_matrix_vector(engine)
                    end

                    env_block_enls[i] = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
                end

                Ψ0_guess = eig_prediction(Ψ[i], sys_label, sys_block_enls[i], env_block_enls[i], sys_trmats[i], env_trmats[i], widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)

                if isroot(runtime)
                    env_block, env_trmats[i] = load_block_and_trmat(storage, env_label, L - sys_blocks[i].length - 2, engine, Val(Nc))

                    env_tensor_dict = spin_operators!(storage, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
                else
                    env_block = Block(L - sys_blocks[i].length - 2, Tuple{Int, Int}[], γ_type[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
                    env_tensor_dict = empty_engine_tensor_dict(engine)
                    env_trmats[i] = empty_engine_matrix_vector(engine)
                end

                env_block_enls[i] = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)

                if verbose && i == 1
                    root_println(runtime, graphic(sys_blocks[i], env_block; sys_label = sys_label))
                end

                result = dmrg_step_result!(SiSj, sys_label, sys_blocks[i], env_block, sys_tensor_dicts[i], env_tensor_dict, sys_block_enls[i], env_block_enls[i], Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); Ψ0_guess = Ψ0_guess, lattice = lattice, alg = alg, noisy = verbose && i == 1)
                sys_blocks[i], sys_tensor_dicts[i], sys_block_enls[i], trerr, energy, Ψ[i], sys_trmats[i], ee, es = result.block, result.tensor_dict, result.block_enl, result.trerr, result.energy, result.Ψ, result.trmat, result.ee, result.es

                _log_phase_result(runtime, verbose && i == 1, energy, L, ee)
            end
        end

        if L == N
            ES = _record_phase_result!(m_list, errors, energies, EEs, m_warmup, result)
        end

        _save_edge_blocks!(storage, mirror, sys_blocks[1], sys_trmats[1], sys_blocks[end], sys_trmats[end], runtime)
    end

    return L, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, ES
end

function _sweep_phase!(SiSj, Ψ, EE, ES, m_list, errors, energies, EEs, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, storage, L, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; Nc, m_sweep_list, m_cooldown, verbose) = config
    (; rank) = runtime

    state = _init_sweep_state(Ψ, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, storage, L, config, runtime, Val(Nc))
    measurement = false

    for m in Iterators.flatten([m_sweep_list, Iterators.repeated(m_cooldown)])
        if verbose
            root_println(runtime, "#")
            if measurement
                root_println(runtime, "# Measurement step with (m, α) = ", m)
            else
                root_println(runtime, "# Performing sweep with (m, α) = ", m)
            end
            root_println(runtime, "#")
        end
        while true
            result = _sweep_step!(SiSj, state, EE, storage, L, m, measurement, config, runtime, Val(Nc))

            if state.sys_label == :l && 2state.sys_block.length == L
                ES = _record_sweep_result!(m_list, errors, energies, EEs, m, result)
                break
            end
        end

        if measurement
            break
        end

        measurement = _update_measurement_flag(measurement, energies, EEs, config, runtime)
    end

    return ES, EE
end
