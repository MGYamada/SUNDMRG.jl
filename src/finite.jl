function _run_DMRG(model::HeisenbergModelSU{Nc}, lattice, Lx, Ly, m_warmup, m_sweep_list, m_cooldown, engine; target = 0, lanczos_maxiter = 100, widthmax = 0, tables = nothing, fileio = false, scratch = ".", ES_max = 20.0, tol_energy = 1e-5, tol_EE = 1e-3, max_cooldown_sweeps = 100, correlation = :none, margin = 0, alg = :slow, verbose = true, manage_mpi = true) where Nc
    _collective_if_active("option validation") do
        _validate_run_DMRG_options(Val(Nc), target, lanczos_maxiter, widthmax, tables, fileio, tol_energy, tol_EE, max_cooldown_sweeps, margin, verbose, manage_mpi)
        correlation ∈ (:none, :nn, :chain) || throw(ArgumentError("correlation must be :none, :nn, or :chain"))
        alg ∈ (:slow, :fast) || throw(ArgumentError("alg must be :slow or :fast"))
    end

    was_initialized = MPI.Initialized()
    return _with_cleanup(
        () -> begin
            if manage_mpi
                init_DMRG!()
            elseif !MPI.Initialized() || MPI.Finalized()
                throw(ArgumentError("MPI must be initialized before run_DMRG(...; manage_mpi = false)"))
            end

            runtime_initialized = false
            rank = 0
            Ncpu = 1
            comm = MPI.COMM_WORLD
            return _with_cleanup(
                () -> begin
                    comm, rank, Ncpu = _comm_context()
                    on_the_fly, mirror, γ_type, γ_list, N, signfactor = _init_runtime_and_engine(engine, lattice, Lx, Ly, Nc, rank, Ncpu)
                    runtime_initialized = true
                    config, runtime = _collective_local(comm, rank, Ncpu, "run configuration") do
                        config = _FiniteRunConfig(Val(lattice), Lx, Ly, N, Nc, m_warmup, m_sweep_list, m_cooldown, target, Int(lanczos_maxiter), widthmax, tables, Val(fileio), scratch, ES_max, tol_energy, tol_EE, Int(max_cooldown_sweeps), Val(correlation), margin, Val(alg), verbose)
                        runtime = _FiniteRuntime(engine, comm, rank, Ncpu, on_the_fly, mirror, γ_type, γ_list, signfactor)
                        (config, runtime)
                    end
                    return _run_DMRG_impl(config, runtime, Val(Nc))
                end,
                () -> begin
                    if runtime_initialized
                        _collective_local(comm, rank, Ncpu, "engine cleanup") do
                            _finalize_runtime!(engine, nothing, rank)
                        end
                    end
                end,
            )
        end,
        () -> begin
            # Detect ownership even if initialization throws after MPI became
            # active. An engine/storage failure must not skip this cleanup.
            if manage_mpi && !was_initialized
                finalize_DMRG!()
            end
        end,
    )
end

function _validate_run_DMRG_options(::Val{Nc}, target, lanczos_maxiter, widthmax, tables, fileio, tol_energy, tol_EE, max_cooldown_sweeps, margin, verbose, manage_mpi) where Nc
    Nc isa Integer && !(Nc isa Bool) || throw(ArgumentError("Nc must be an integer"))
    Nc >= 2 || throw(ArgumentError("Nc must be at least 2"))
    target isa Integer || throw(ArgumentError("target must be an integer"))
    target >= 0 || throw(ArgumentError("target must be nonnegative"))
    lanczos_maxiter isa Integer && !(lanczos_maxiter isa Bool) || throw(ArgumentError("lanczos_maxiter must be an integer"))
    lanczos_maxiter > 0 || throw(ArgumentError("lanczos_maxiter must be positive"))
    target + 1 <= lanczos_maxiter || throw(ArgumentError("target + 1 must not exceed lanczos_maxiter"))
    widthmax isa Integer || throw(ArgumentError("widthmax must be an integer"))
    widthmax >= 0 || throw(ArgumentError("widthmax must be nonnegative"))
    margin isa Integer || throw(ArgumentError("margin must be an integer"))
    margin >= 0 || throw(ArgumentError("margin must be nonnegative"))
    max_cooldown_sweeps isa Integer || throw(ArgumentError("max_cooldown_sweeps must be an integer"))
    max_cooldown_sweeps > 0 || throw(ArgumentError("max_cooldown_sweeps must be positive"))
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

function _run_DMRG_impl(config::_FiniteRunConfig, runtime::_FiniteRuntime, ::Val{Nc}) where Nc
    state = _init_state(config, runtime)

    return _with_cleanup(
        () -> begin
            _warmup_phase!(state, config, runtime)

            growth = _growth_phase!(state, config, runtime)

            state.ES, state.EE = _sweep_phase!(state.SiSj, state.Ψ, state.EE, state.ES, state.m_list, state.errors, state.energies, state.EEs, growth.sys_blocks, growth.sys_tensor_dicts, growth.sys_trmats, growth.sys_block_enls, state.storage, growth.L, config, runtime)

            return _collective_local(runtime, "result construction") do
                ESrtn = Dict{NTuple{Nc, Int}, Vector{Float64}}()
                for (key, value) in state.ES
                    ESrtn[weight(key)] = value
                end

                if Nc == 2
                    map!(x -> 0.5x, values(state.SiSj))
                end

                (runtime.rank, runtime.rank == 0 ? DMRGOutput(state.m_list, state.errors, state.energies, state.EEs, state.EE, ESrtn, state.SiSj) : nothing)
            end
        end,
        () -> begin
            _collective_local(runtime, "storage cleanup") do
                if isroot(runtime)
                    cleanup_storage!(state.storage)
                end
            end
        end,
    )
end
