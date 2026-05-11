mutable struct _SweepState{ΨT,SjT,SysBlockT,EnvBlockT,SysTensorT,EnvTensorT,SysTrmatT,EnvTrmatT,SysEnlT,EnvEnlT}
    Ψ0::ΨT
    Sj::SjT
    sys_label::Symbol
    env_label::Symbol
    sys_block::SysBlockT
    env_block::EnvBlockT
    sys_tensor_dict::SysTensorT
    env_tensor_dict::EnvTensorT
    sys_trmat::SysTrmatT
    env_trmat::EnvTrmatT
    sys_block_enl::SysEnlT
    env_block_enl::EnvEnlT
end

function _load_sweep_env(storage, env_label, env_len, config::_FiniteRunConfig, runtime::_FiniteRuntime, ::Val{Nc}) where Nc
    (; lattice, Ly, widthmax, tables) = config
    (; engine, comm, rank, Ncpu, on_the_fly, γ_type, signfactor) = runtime

    if isroot(runtime)
        env_block, env_trmat = load_block_and_trmat(storage, env_label, env_len, engine, Val(Nc))
        env_tensor_dict = spin_operators!(storage, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    else
        env_block = Block(env_len, Tuple{Int, Int}[], γ_type[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
        env_tensor_dict = empty_engine_tensor_dict(engine)
        env_trmat = empty_engine_matrix_vector(engine)
    end

    env_block_enl = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    return env_block, env_tensor_dict, env_trmat, env_block_enl
end

function _init_sweep_state(Ψ, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, storage, L, config::_FiniteRunConfig, runtime::_FiniteRuntime, ::Val{Nc}) where Nc
    sys_label, env_label = :l, :r
    sys_block = sys_blocks[1]
    env_block, env_tensor_dict, env_trmat, env_block_enl = _load_sweep_env(storage, env_label, L - sys_block.length - 1, config, runtime, Val(Nc))

    _SweepState(
        Ψ[1],
        Matrix{Vector{Matrix{Float64}}}(undef, 0, 0),
        sys_label,
        env_label,
        sys_block,
        env_block,
        sys_tensor_dicts[1],
        env_tensor_dict,
        sys_trmats[1],
        env_trmat,
        sys_block_enls[1],
        env_block_enl,
    )
end

function _reverse_sweep_end!(state::_SweepState, Ψ0_guess, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; widthmax, tables) = config
    (; comm, rank, Ncpu, on_the_fly, γ_list, engine) = runtime

    if state.env_block.length == 0
        Ψ0_guess = wavefunction_reverse(Ψ0_guess, state.sys_label, state.sys_block_enl, state.env_block_enl, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)

        state.sys_block, state.env_block = state.env_block, state.sys_block
        state.sys_tensor_dict, state.env_tensor_dict = state.env_tensor_dict, state.sys_tensor_dict
        state.sys_trmat, state.env_trmat = state.env_trmat, state.sys_trmat
        state.sys_block_enl, state.env_block_enl = state.env_block_enl, state.sys_block_enl
        state.sys_label, state.env_label = state.env_label, state.sys_label
    end

    return Ψ0_guess
end

function _sweep_step!(SiSj, state::_SweepState, EE, storage, L, m, measurement, config::_FiniteRunConfig, runtime::_FiniteRuntime, ::Val{Nc}) where Nc
    (; lattice, Ly, widthmax, target, tables, alg, correlation, margin, ES_max, verbose) = config
    (; engine, comm, rank, Ncpu, on_the_fly, γ_list, signfactor) = runtime

    Ψ0_guess = eig_prediction(state.Ψ0, state.sys_label, state.sys_block_enl, state.env_block_enl, state.sys_trmat, state.env_trmat, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)

    state.env_block, state.env_tensor_dict, state.env_trmat, state.env_block_enl = _load_sweep_env(storage, state.env_label, L - state.sys_block.length - 2, config, runtime, Val(Nc))
    Ψ0_guess = _reverse_sweep_end!(state, Ψ0_guess, config, runtime)

    if verbose
        root_println(runtime, graphic(state.sys_block, state.env_block; sys_label = state.sys_label))
    end

    cor = measurement && (state.sys_label == :r || state.sys_block.length == 0) ? correlation : Val(:none)
    result = dmrg_step_result!(SiSj, state.sys_label, state.sys_block, state.env_block, state.sys_tensor_dict, state.env_tensor_dict, state.sys_block_enl, state.env_block_enl, Ly, m..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); Ψ0_guess = Ψ0_guess, ES_max = ES_max, correlation = cor, margin = margin, lattice = lattice, Sj = state.Sj, alg = alg, noisy = verbose)
    state.sys_block = result.block
    state.sys_tensor_dict = result.tensor_dict
    state.sys_block_enl = result.block_enl
    state.Ψ0 = result.Ψ
    state.sys_trmat = result.trmat
    state.Sj = result.Sj

    if isroot(runtime)
        if state.sys_label == :r && state.env_block_enl.length % Ly == 0
            EE[state.env_block_enl.length ÷ Ly] = result.ee
        end
        if verbose
            root_println(runtime, "E / N = ", result.energy / L)
            root_println(runtime, "E     = ", result.energy)
            root_println(runtime, "S_EE  = ", result.ee)
        end
        save_block_and_trmat!(storage, state.sys_label, state.sys_block, state.sys_trmat)
    end

    return result
end

function _update_measurement_flag(measurement, energies, EEs, config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; m_sweep_list, tol_energy, tol_EE) = config
    (; comm, rank) = runtime

    if length(energies) <= length(m_sweep_list) + 1
        return measurement
    end

    if isroot(runtime)
        measurement = abs((energies[end] - energies[end - 1]) / energies[end]) < tol_energy && abs((EEs[end] - EEs[end - 1]) / EEs[end]) < tol_EE
        return MPI.bcast(measurement, 0, comm)::Bool
    end

    MPI.bcast(nothing, 0, comm)::Bool
end
