function _run_DMRG(model::HeisenbergModelSU{Nc}, lattice, Lx, Ly, m_warmup, m_sweep_list, m_cooldown, engine; target = 0, widthmax = 0, tables = nothing, fileio = false, scratch = ".", ES_max = 20.0, tol_energy = 1e-5, tol_EE = 1e-3, correlation = :none, margin = 0, alg = :slow, verbose = true, manage_mpi = true) where Nc
    @assert correlation == :none || correlation == :nn || correlation == :chain
    @assert alg == :slow || alg == :fast

    if manage_mpi
        init_DMRG!()
    elseif !MPI.Initialized() || MPI.Finalized()
        throw(ArgumentError("MPI must be initialized before run_DMRG(...; manage_mpi = false)"))
    end
    comm, rank, Ncpu = _comm_context()
    on_the_fly, mirror, γ_type, γ_list, N, signfactor = _init_runtime_and_engine(engine, lattice, Lx, Ly, Nc, rank, Ncpu)
    config = _FiniteRunConfig(lattice, Lx, Ly, N, Nc, m_warmup, m_sweep_list, m_cooldown, target, widthmax, tables, fileio, scratch, ES_max, tol_energy, tol_EE, correlation, margin, alg, verbose)
    runtime = _FiniteRuntime(engine, comm, rank, Ncpu, on_the_fly, mirror, γ_type, γ_list, signfactor)

    m_list, errors, energies, EEs, EE, ES, SiSj, storage, blockL, blockL_tensor_dict, blockR, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ = _init_state(config, runtime)

    blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ, ES = _warmup_phase!(SiSj, blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ, m_list, errors, energies, EEs, ES, storage, config, runtime)

    L, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, ES = _growth_phase!(SiSj, blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, trmatL, trmatR, Ψ, m_list, errors, energies, EEs, ES, storage, config, runtime)

    ES, EE = _sweep_phase!(SiSj, Ψ, EE, ES, m_list, errors, energies, EEs, sys_blocks, sys_tensor_dicts, sys_trmats, sys_block_enls, storage, L, config, runtime)

    _finalize_runtime!(engine, storage, rank)
    if manage_mpi
        finalize_DMRG!()
    end

    ESrtn = Dict{NTuple{Nc, Int}, Vector{Float64}}()
    for (key, value) in ES
        ESrtn[weight(key)] = value
    end

    if Nc == 2
        map!(x -> 0.5x, values(SiSj))
    end

    rank, DMRGOutput(m_list, errors, energies, EEs, EE, ESrtn, SiSj)
end
