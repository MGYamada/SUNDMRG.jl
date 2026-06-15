function _run_DMRG(model::HeisenbergModelSU{Nc}, lattice, Lx, Ly, m_warmup, m_sweep_list, m_cooldown, engine; target = 0, widthmax = 0, tables = nothing, fileio = false, scratch = ".", ES_max = 20.0, tol_energy = 1e-5, tol_EE = 1e-3, correlation = :none, margin = 0, alg = :slow, verbose = true, manage_mpi = true) where Nc
    _validate_run_DMRG_options(Val(Nc), target, widthmax, tables, fileio, tol_energy, tol_EE, margin, verbose, manage_mpi)
    correlation ∈ (:none, :nn, :chain) || throw(ArgumentError("correlation must be :none, :nn, or :chain"))
    alg ∈ (:slow, :fast) || throw(ArgumentError("alg must be :slow or :fast"))

    did_init = Ref(false)
    runtime_initialized = false
    runtime_finalized = Ref(false)
    rank = 0
    if manage_mpi
        was_initialized = MPI.Initialized()
        try
            did_init[] = init_DMRG!()
        catch
            if !was_initialized && MPI.Initialized() && !MPI.Finalized()
                finalize_DMRG!()
            end
            rethrow()
        end
    elseif !MPI.Initialized() || MPI.Finalized()
        throw(ArgumentError("MPI must be initialized before run_DMRG(...; manage_mpi = false)"))
    end

    try
        comm, rank, Ncpu = _comm_context()
        on_the_fly, mirror, γ_type, γ_list, N, signfactor = _init_runtime_and_engine(engine, lattice, Lx, Ly, Nc, rank, Ncpu)
        runtime_initialized = true
        config = _FiniteRunConfig(Val(lattice), Lx, Ly, N, Nc, m_warmup, m_sweep_list, m_cooldown, target, widthmax, tables, Val(fileio), scratch, ES_max, tol_energy, tol_EE, Val(correlation), margin, Val(alg), verbose)
        runtime = _FiniteRuntime(engine, comm, rank, Ncpu, on_the_fly, mirror, γ_type, γ_list, signfactor)

        return _run_DMRG_impl(config, runtime, runtime_finalized, Val(Nc))
    finally
        if runtime_initialized && !runtime_finalized[]
            _finalize_runtime!(engine, nothing, rank)
        end
        if manage_mpi && did_init[]
            finalize_DMRG!()
        end
    end
end

function _validate_run_DMRG_options(::Val{Nc}, target, widthmax, tables, fileio, tol_energy, tol_EE, margin, verbose, manage_mpi) where Nc
    target isa Integer || throw(ArgumentError("target must be an integer"))
    target >= 0 || throw(ArgumentError("target must be nonnegative"))
    widthmax isa Integer || throw(ArgumentError("widthmax must be an integer"))
    widthmax >= 0 || throw(ArgumentError("widthmax must be nonnegative"))
    margin isa Integer || throw(ArgumentError("margin must be an integer"))
    margin >= 0 || throw(ArgumentError("margin must be nonnegative"))
    fileio isa Bool || throw(ArgumentError("fileio must be true or false"))
    verbose isa Bool || throw(ArgumentError("verbose must be true or false"))
    manage_mpi isa Bool || throw(ArgumentError("manage_mpi must be true or false"))

    _positive_finite_option(tol_energy, "tol_energy")
    _positive_finite_option(tol_EE, "tol_EE")

    if Nc > 2
        widthmax > 0 || throw(ArgumentError("widthmax must be positive for SU(N > 2)"))
        tables === nothing && throw(ArgumentError("tables must be provided for SU(N > 2)"))
    end

    return nothing
end

function _positive_finite_option(value, name)
    value isa Real || throw(ArgumentError("$name must be real"))
    value = Float64(value)
    isfinite(value) && value > 0.0 || throw(ArgumentError("$name must be positive and finite"))
    return value
end

function _run_DMRG_impl(config::_FiniteRunConfig, runtime::_FiniteRuntime, runtime_finalized, ::Val{Nc}) where Nc
    state = _init_state(config, runtime)

    try
        _warmup_phase!(state, config, runtime)

        growth = _growth_phase!(state, config, runtime)

        state.ES, state.EE = _sweep_phase!(state.SiSj, state.Ψ, state.EE, state.ES, state.m_list, state.errors, state.energies, state.EEs, growth.sys_blocks, growth.sys_tensor_dicts, growth.sys_trmats, growth.sys_block_enls, state.storage, growth.L, config, runtime)

        ESrtn = Dict{NTuple{Nc, Int}, Vector{Float64}}()
        for (key, value) in state.ES
            ESrtn[weight(key)] = value
        end

        if Nc == 2
            map!(x -> 0.5x, values(state.SiSj))
        end

        return runtime.rank, runtime.rank == 0 ? DMRGOutput(state.m_list, state.errors, state.energies, state.EEs, state.EE, ESrtn, state.SiSj) : nothing
    finally
        _finalize_runtime!(runtime.engine, state.storage, runtime.rank)
        runtime_finalized[] = true
    end
end
