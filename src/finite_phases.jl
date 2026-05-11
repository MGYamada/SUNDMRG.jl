function _record_step_result!(m_list, errors, energies, EEs, m, result::AbstractDMRGStepResult)
    push!(m_list, m)
    push!(errors, result.trerr)
    push!(energies, result.energy)
    push!(EEs, result.ee)
    return result.es
end

function _apply_left_step_result!(state::_FiniteState, result::AbstractDMRGStepResult)
    state.blockL = result.block
    state.blockL_tensor_dict = result.tensor_dict
    state.blockL_enl = result.block_enl
    state.Ψ[1] = result.Ψ
    state.trmatL = result.trmat
    return result.es
end

function _apply_mirror_env_result!(state::_FiniteState, result::DMRGStepResultWithEnv, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; widthmax, tables) = config
    (; comm, rank, Ncpu, on_the_fly, γ_list, engine) = runtime

    state.blockR = result.env_block
    state.blockR_tensor_dict = result.env_tensor_dict
    state.blockR_enl = result.env_block_enl
    state.trmatR = result.env_trmat
    state.Ψ[2] = wavefunction_reverse(state.Ψ[1], :l, state.blockL, state.blockR, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
    return nothing
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

_worker_empty_block(len, γ_type) = Block(len, Tuple{Int, Int}[], γ_type[], Int[], Int[], ScalarDictCPU())

function _worker_empty_environment(len, γ_type, engine)
    block = _worker_empty_block(len, γ_type)
    tensor_dict = empty_engine_tensor_dict(engine)
    trmat = empty_engine_matrix_vector(engine)
    return block, tensor_dict, trmat
end

function _init_state(config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; lattice, Lx, Ly, Nc, target, m_warmup, widthmax, tables, fileio, scratch, verbose) = config
    (; engine, on_the_fly, mirror, γ_type, comm, rank, Ncpu, signfactor) = runtime

    storage = nothing
    m_list = Tuple{Int, Float64}[]
    errors = Float64[]
    energies = Float64[]
    EEs = Float64[]
    EE = fill(NaN, Lx - 1)
    ES = Dict{SUNIrrep{Nc}, Vector{Float64}}()
    SiSj = Dict{Tuple{Int, Int}, Float64}()

    block_table = Dict{Tuple{Symbol, Int}, Block{Nc}}()
    tensor_table = Dict{Tuple{Symbol, Int, Int}, Matrix{Vector{Matrix{Float64}}}}()
    trmat_table = Dict{Tuple{Symbol, Int}, Vector{Matrix{Float64}}}()

    try
        if isroot(runtime)
            storage, blockL, blockL_tensor_dict, trmatL, blockR, blockR_tensor_dict, trmatR =
                _init_root_state_edges(config, runtime, block_table, trmat_table, tensor_table, Val(Nc))
        else
            storage, blockL, blockL_tensor_dict, trmatL, blockR, blockR_tensor_dict, trmatR =
                _init_worker_state_edges(config, runtime, block_table, trmat_table, tensor_table, Val(Nc))
        end

        blockL_enl = enlarge_block(blockL, blockL_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
        if !mirror
            blockR_enl = enlarge_block(blockR, blockR_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
        else
            blockR_enl = blockL_enl
        end

        Ψ = empty_engine_tensor_matrices(engine, mirror ? 1 : 2)

        return _FiniteState{Nc,typeof(storage),typeof(blockL),typeof(blockL_tensor_dict),typeof(blockR),typeof(blockR_tensor_dict),typeof(blockL_enl),typeof(blockR_enl),typeof(trmatL),typeof(trmatR),typeof(Ψ)}(
            m_list,
            errors,
            energies,
            EEs,
            EE,
            ES,
            SiSj,
            storage,
            blockL,
            blockL_tensor_dict,
            blockR,
            blockR_tensor_dict,
            blockL_enl,
            blockR_enl,
            trmatL,
            trmatR,
            Ψ,
        )
    catch
        if storage !== nothing && isroot(runtime)
            cleanup_storage!(storage)
        end
        rethrow()
    end
end

function _init_root_state_edges(config::_FiniteRunConfig, runtime::_FiniteRuntime, block_table, trmat_table, tensor_table, ::Val{Nc}) where Nc
    (; lattice, Lx, Ly, target, m_warmup, widthmax, fileio, scratch, verbose) = config
    (; engine, on_the_fly, mirror, rank) = runtime

    if verbose
        root_println(runtime, repeat("-", 60))
        root_println(runtime, "SU($Nc) DMRG simulation on the $Lx x $Ly $(_lattice_name(lattice)) lattice cylinder:")
        if _on_the_fly(on_the_fly)
            root_println(runtime, "All representations are used in the calculation.")
        else
            irreps = irreplist(Nc, widthmax)
            root_println(runtime, length(irreps), " irreps from ", weight(first(irreps)), " to ", weight(last(irreps)), " are used in the calculation.")
        end
        root_println(runtime, target == 0 ? "The ground state" : "The excited state #$target", " will be calculated.")
        root_println(runtime, repeat("-", 60))
    end

    storage = init_internal_storage(fileio, scratch, block_table, trmat_table, tensor_table, rank)
    blockL = Block(0, Tuple{Int, Int}[], [trivialirrep(Val(Nc))], [1], [1], ScalarDictCPU(:H => [zeros(1, 1)]))
    blockL_tensor_dict = TensorDictCPU()
    trmatL = [to_engine_array(engine, diagm([1.0]))]

    if !mirror
        blockR = Block(0, Tuple{Int, Int}[], [trivialirrep(Val(Nc))], [1], [1], ScalarDictCPU(:H => [zeros(1, 1)]))
        blockR_tensor_dict = TensorDictCPU()
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

    return storage, blockL, blockL_tensor_dict, trmatL, blockR, blockR_tensor_dict, trmatR
end

function _init_worker_state_edges(config::_FiniteRunConfig, runtime::_FiniteRuntime, block_table, trmat_table, tensor_table, ::Val{Nc}) where Nc
    (; fileio, scratch) = config
    (; engine, mirror, γ_type, rank) = runtime
    storage = init_internal_storage(fileio, scratch, block_table, trmat_table, tensor_table, rank)
    blockL, blockL_tensor_dict, trmatL = _worker_empty_environment(0, γ_type, engine)

    if !mirror
        blockR, blockR_tensor_dict, trmatR = _worker_empty_environment(0, γ_type, engine)
    else
        blockR = blockL
        blockR_tensor_dict = blockL_tensor_dict
        trmatR = empty_engine_matrix_vector(engine)
    end

    return storage, blockL, blockL_tensor_dict, trmatL, blockR, blockR_tensor_dict, trmatR
end

function _warmup_phase!(state::_FiniteState{Nc}, config::_FiniteRunConfig, runtime::_FiniteRuntime) where Nc
    (; Ly, N, m_warmup, verbose) = config
    (; mirror) = runtime

    while state.blockL.length < Ly
        if verbose && isroot(runtime)
            if mirror
                root_println(runtime, graphic(state.blockL, state.blockL))
            else
                root_println(runtime, graphic(state.blockL, state.blockR))
            end
        end

        if mirror
            blocks = _dmrg_step_blocks(:l, state.blockL, state.blockL, state.blockL_tensor_dict, state.blockL_tensor_dict, state.blockL_enl, state.blockL_enl)
            request = _dmrg_step_request(blocks, m_warmup, Val(false); noisy = verbose)
            result = dmrg_step_result!(state.SiSj, request, config, runtime, Val(Nc))
            _apply_left_step_result!(state, result)
        else
            blocks = _dmrg_step_blocks(:l, state.blockL, state.blockR, state.blockL_tensor_dict, state.blockR_tensor_dict, state.blockL_enl, state.blockR_enl)
            request = _dmrg_step_request(blocks, m_warmup, Val(true); noisy = verbose)
            result = dmrg_step_result!(state.SiSj, request, config, runtime, Val(Nc))
            _apply_left_step_result!(state, result)
            _apply_mirror_env_result!(state, result, config, runtime)
        end

        if 2state.blockL.length == N
            state.ES = _record_step_result!(state.m_list, state.errors, state.energies, state.EEs, m_warmup, result)
        end

        _log_phase_result(runtime, verbose, result.energy, 2state.blockL.length, result.ee)
        _save_edge_blocks!(state.storage, mirror, state.blockL, state.trmatL, state.blockR, state.trmatR, runtime)
    end

    return state
end

function _init_growth_workspaces(state::_FiniteState, mirror)
    if mirror
        sys_blocks = [state.blockL]
        sys_tensor_dicts = [state.blockL_tensor_dict]
        sys_trmats = [state.trmatL]
        sys_block_enls = [state.blockL_enl]
        env_trmats = [state.trmatL]
        env_block_enls = [state.blockL_enl]
    else
        sys_blocks = [state.blockL, state.blockR]
        sys_tensor_dicts = [state.blockL_tensor_dict, state.blockR_tensor_dict]
        sys_trmats = [state.trmatL, state.trmatR]
        sys_block_enls = [state.blockL_enl, state.blockR_enl]
        env_trmats = [state.trmatR, state.trmatL]
        env_block_enls = [state.blockR_enl, state.blockL_enl]
    end

    return sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, env_trmats, env_block_enls
end

_growth_side_count(mirror) = mirror ? 1 : 2
_growth_side_labels(i) = i == 1 ? (:l, :r) : (:r, :l)

function _growth_frontier_graphic(sys_blocks, mirror)
    if mirror
        return graphic(sys_blocks[1], sys_blocks[1]; sys_label = :l)
    end
    return graphic(sys_blocks[1], sys_blocks[2]; sys_label = :l)
end

function _apply_growth_sys_result!(state::_FiniteState, i, result::AbstractDMRGStepResult, sys_blocks, sys_tensor_dicts, sys_block_enls, sys_trmats)
    sys_blocks[i] = result.block
    sys_tensor_dicts[i] = result.tensor_dict
    sys_block_enls[i] = result.block_enl
    state.Ψ[i] = result.Ψ
    sys_trmats[i] = result.trmat
    return nothing
end

function _reverse_growth_wavefunction!(state::_FiniteState, sys_blocks, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; widthmax, tables) = config
    (; comm, rank, Ncpu, on_the_fly, γ_list, engine) = runtime
    state.Ψ[2] = wavefunction_reverse(state.Ψ[1], :l, sys_blocks[1], sys_blocks[2], widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
    return nothing
end

function _run_growth_frontier_step!(state::_FiniteState{Nc}, sys_blocks, sys_tensor_dicts, sys_block_enls, sys_trmats, config::_FiniteRunConfig, runtime::_FiniteRuntime) where Nc
    (; m_warmup, verbose) = config
    (; mirror) = runtime

    if mirror
        blocks = _dmrg_step_blocks(:l, sys_blocks[1], sys_blocks[1], sys_tensor_dicts[1], sys_tensor_dicts[1], sys_block_enls[1], sys_block_enls[1])
        request = _dmrg_step_request(blocks, m_warmup, Val(false); noisy = verbose)
        result = dmrg_step_result!(state.SiSj, request, config, runtime, Val(Nc))
        _apply_growth_sys_result!(state, 1, result, sys_blocks, sys_tensor_dicts, sys_block_enls, sys_trmats)
    else
        blocks = _dmrg_step_blocks(:l, sys_blocks[1], sys_blocks[2], sys_tensor_dicts[1], sys_tensor_dicts[2], sys_block_enls[1], sys_block_enls[2])
        request = _dmrg_step_request(blocks, m_warmup, Val(true); noisy = verbose)
        result = dmrg_step_result!(state.SiSj, request, config, runtime, Val(Nc))
        _apply_growth_sys_result!(state, 1, result, sys_blocks, sys_tensor_dicts, sys_block_enls, sys_trmats)
        sys_blocks[2] = result.env_block
        sys_tensor_dicts[2] = result.env_tensor_dict
        sys_block_enls[2] = result.env_block_enl
        sys_trmats[2] = result.env_trmat
        _reverse_growth_wavefunction!(state, sys_blocks, config, runtime)
    end

    return result
end

function _load_growth_environment(state::_FiniteState, env_label, env_len, config::_FiniteRunConfig, runtime::_FiniteRuntime, ::Val{Nc}) where Nc
    (; lattice, Ly, widthmax, tables) = config
    (; engine, comm, rank, Ncpu, on_the_fly, γ_type, signfactor) = runtime

    if isroot(runtime)
        env_block, env_trmat = load_block_and_trmat(state.storage, env_label, env_len, engine, Val(Nc))
        env_tensor_dict = spin_operators!(state.storage, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    else
        env_block, env_tensor_dict, env_trmat = _worker_empty_environment(env_len, γ_type, engine)
    end

    return env_block, env_tensor_dict, env_trmat
end

function _enlarge_growth_environment(env_block, env_tensor_dict, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; lattice, Ly, widthmax, tables) = config
    (; engine, comm, rank, Ncpu, on_the_fly, signfactor) = runtime
    return enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
end

function _refresh_growth_prediction_environment!(state::_FiniteState{Nc}, i, env_label, L, sys_blocks, env_trmats, env_block_enls, config::_FiniteRunConfig, runtime::_FiniteRuntime) where Nc
    env_len = L - sys_blocks[i].length - 1
    env_block, env_tensor_dict, env_trmats[i] = _load_growth_environment(state, env_label, env_len, config, runtime, Val(Nc))
    env_block_enls[i] = _enlarge_growth_environment(env_block, env_tensor_dict, config, runtime)
    return nothing
end

function _growth_eigenvector_prediction(state::_FiniteState, i, sys_label, sys_block_enls, sys_trmats, env_trmats, env_block_enls, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; widthmax, tables) = config
    (; comm, rank, Ncpu, on_the_fly, γ_list, engine) = runtime
    return eig_prediction(state.Ψ[i], sys_label, sys_block_enls[i], env_block_enls[i], sys_trmats[i], env_trmats[i], widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
end

function _run_growth_inner_step!(state::_FiniteState{Nc}, i, L, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, env_trmats, env_block_enls, config::_FiniteRunConfig, runtime::_FiniteRuntime) where Nc
    (; Ly, m_warmup, verbose) = config
    sys_label, env_label = _growth_side_labels(i)

    if sys_blocks[i].length % Ly == 0
        _refresh_growth_prediction_environment!(state, i, env_label, L, sys_blocks, env_trmats, env_block_enls, config, runtime)
    end

    Ψ0_guess = _growth_eigenvector_prediction(state, i, sys_label, sys_block_enls, sys_trmats, env_trmats, env_block_enls, config, runtime)

    env_len = L - sys_blocks[i].length - 2
    env_block, env_tensor_dict, env_trmats[i] = _load_growth_environment(state, env_label, env_len, config, runtime, Val(Nc))
    env_block_enls[i] = _enlarge_growth_environment(env_block, env_tensor_dict, config, runtime)

    if verbose && i == 1
        root_println(runtime, graphic(sys_blocks[i], env_block; sys_label = sys_label))
    end

    blocks = _dmrg_step_blocks(sys_label, sys_blocks[i], env_block, sys_tensor_dicts[i], env_tensor_dict, sys_block_enls[i], env_block_enls[i])
    request = _dmrg_step_request(blocks, m_warmup, Val(false); Ψ0_guess = Ψ0_guess, noisy = verbose && i == 1)
    result = dmrg_step_result!(state.SiSj, request, config, runtime, Val(Nc))
    _apply_growth_sys_result!(state, i, result, sys_blocks, sys_tensor_dicts, sys_block_enls, sys_trmats)

    _log_phase_result(runtime, verbose && i == 1, result.energy, L, result.ee)
    return result
end

function _growth_phase!(state::_FiniteState{Nc}, config::_FiniteRunConfig, runtime::_FiniteRuntime) where Nc
    (; Ly, N, verbose) = config
    (; mirror) = runtime

    L = 2state.blockL.length
    sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, env_trmats, env_block_enls = _init_growth_workspaces(state, mirror)

    while L < N
        record_result = nothing
        if sys_block_enls[1].length % Ly == 0
            if verbose && isroot(runtime)
                root_println(runtime, _growth_frontier_graphic(sys_blocks, mirror))
            end

            L = 2sys_block_enls[1].length
            result = _run_growth_frontier_step!(state, sys_blocks, sys_tensor_dicts, sys_block_enls, sys_trmats, config, runtime)
            record_result = result

            _log_phase_result(runtime, verbose, result.energy, L, result.ee)
        else
            for i in 1 : _growth_side_count(mirror)
                result = _run_growth_inner_step!(state, i, L, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, env_trmats, env_block_enls, config, runtime)
                if i == 1
                    record_result = result
                end
            end
        end

        if L == N
            state.ES = _record_step_result!(state.m_list, state.errors, state.energies, state.EEs, m_warmup, record_result)
        end

        _save_edge_blocks!(state.storage, mirror, sys_blocks[1], sys_trmats[1], sys_blocks[end], sys_trmats[end], runtime)
    end

    return _GrowthState(L, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls)
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
                ES = _record_step_result!(m_list, errors, energies, EEs, m, result)
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
