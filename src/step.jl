"""
    dmrg_step!(...)

A single DMRG step, returned as a `DMRGStepResult` with explicit fields.
"""
function dmrg_step!(SiSj, sys_label, sys::Block{Nc}, env::Block{Nc}, sys_tensor_dict, env_tensor_dict, sys_enl::EnlargedBlock{Nc}, env_enl::EnlargedBlock{Nc}, Ly, m, α, widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, ::Val{env_calc}; Ψ0_guess = nothing, ES_max = -Inf, correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0), alg = :slow, noisy = true) where {Nc, env_calc}
    workspace = _prepare_step_workspace(sys, env, sys_tensor_dict, env_tensor_dict, sys_enl, env_enl, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, lattice, Val(Nc))
    (; sys_αs, env_αs, sys_βs, env_βs, sys_ms, env_ms, sys_len, env_len, superblock_bonds, bonds_hold, x_conn, y_conn, sys_dp, env_dp, sys_enlarge, env_enlarge, OM, superblock_H1, superblock_H2, sys_connS, env_connS, sys_tensor_dict_hold, env_tensor_dict_hold) = workspace

    Ψ0 = _initial_step_wavefunction(Ψ0_guess, env_ms, sys_ms, OM, Ncpu, rank, engine)
    lanczos_context = _StepLanczosContext(target, comm, rank, engine, alg, superblock_H1, bonds_hold, x_conn, y_conn, sys_connS, env_connS, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, sys_enlarge, env_enlarge, sys_tensor_dict_hold, env_tensor_dict_hold, superblock_H2, OM, sys_len, env_len, Ncpu)
    E, time_Lanczos = _run_step_lanczos!(Ψ0, lanczos_context)
    energy = _step_energy(E, time_Lanczos, sys_enl, env_enl, superblock_bonds, comm, rank, noisy, Val(Nc))
    density_context = _StepDensityContext(comm, rank, Ncpu, engine, α, m, ES_max, noisy)
    measurement_context = _StepMeasurementContext(SiSj, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, comm, rank, Ncpu, engine, correlation, margin, lattice)
    block_context = _StepBlockContext(Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine, lattice)
    correction_context = _StepCorrectionContext(superblock_bonds, sys_connS, env_connS, sys_enl, env_enl, x_conn, y_conn, sys_tensor_dict_hold, env_tensor_dict_hold, sys_tensor_dict, env_tensor_dict, sys_ms, env_ms, sys_dp, env_dp, sys_βs, env_βs, sys_enlarge, env_enlarge, engine)

    newblock, newblock_enl, trmat, newtensor_dict = _step_result_buffers(Val(Nc), engine)

    ee = Float64[]
    es = Dict{SUNIrrep{Nc}, Vector{Float64}}[]
    truncation_error = Float64[]

    for switch in 1 : (env_calc ? 2 : 1)
        side = _step_side_context(switch, sys_label, sys_len, env_len, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, x_conn, y_conn, sys, env, sys_enl, env_enl)
        (; len, ms, dp, conn, label, block, block_enl) = side
        βs = side.betas

        balancer = _density_matrix_balancer(ms, density_context.Ncpu, density_context.rank)
        MPI.Bcast!(balancer, 0, comm)

        dimβ = dim.(βs)
        ρs = _reduced_density_matrices(Ψ0, switch, side, dimβ, sys_len, env_len, density_context)

        if density_context.rank == 0 && density_context.α != 0.0
            _apply_density_matrix_correction!(ρs, switch, side, density_context, correction_context)
        end

        MPI.Barrier(comm)

        ρnew = _balanced_density_matrices(ρs, balancer, side, density_context)

        MPI.Barrier(comm)

        λζtemp, time_DM = _density_eigendecomposition(ρnew, density_context)

        time2 = MPI.Reduce(time_DM, MPI.MAX, 0, comm)

        if noisy && rank == 0 && switch == 1
            println(time2, " seconds elapsed in the density matrix diagonalization")
        end

        MPI.Barrier(comm)

        λ, ζ = _collect_density_eigenpairs(λζtemp, balancer, side, density_context)

        MPI.Barrier(comm)

        ee_value, esi, transformation_matrix, msnew, Hnew, indices = _density_truncation_basis(λ, ζ, side, dimβ, sys_enl, env_enl, density_context, Val(Nc), switch)
        push!(ee, ee_value)
        push!(trmat, transformation_matrix)
        push!(es, esi)

        MPI.Barrier(comm)
        tensor_dict, Sj = _measure_step_tensor_dict!(measurement_context, switch, label, sys, env, sys_enl, env_enl, Ψ0, transformation_matrix, Sj)
        MPI.Barrier(comm)

        _push_step_block!(newblock, newtensor_dict, newblock_enl, tensor_dict, side, msnew, Hnew, block_context, Val(Nc))

        if rank == 0
            push!(truncation_error, _step_truncation_error(λ, indices, dimβ, newblock_enl[switch], Val(Nc)))
            if noisy && switch == 1
                println("truncation error: ", truncation_error[switch])
            end
        else
            push!(truncation_error, 0.0)
        end
    end

    _report_overlap(Ψ0_guess, Ψ0, comm, rank, noisy)
    return _dmrg_step_result(
        Val(env_calc),
        newblock[1],
        newtensor_dict[1],
        newblock_enl[1],
        truncation_error[1],
        energy,
        Ψ0,
        trmat[1],
        ee[1],
        es[1],
        Sj,
        newblock,
        newtensor_dict,
        newblock_enl,
        trmat,
    )
end

abstract type AbstractDMRGStepResult end

struct DMRGStepResult{B,T,BEnl,ΨT,TrT,EST,SJT} <: AbstractDMRGStepResult
    block::B
    tensor_dict::T
    block_enl::BEnl
    trerr::Float64
    energy::Float64
    Ψ::ΨT
    trmat::TrT
    ee::Float64
    es::EST
    Sj::SJT
end

struct DMRGStepResultWithEnv{B,T,BEnl,ΨT,TrT,EST,EB,ET,EBEnl,ETrT} <: AbstractDMRGStepResult
    block::B
    tensor_dict::T
    block_enl::BEnl
    trerr::Float64
    energy::Float64
    Ψ::ΨT
    trmat::TrT
    ee::Float64
    es::EST
    env_block::EB
    env_tensor_dict::ET
    env_block_enl::EBEnl
    env_trmat::ETrT
end

function _dmrg_step_result(::Val{false}, block, tensor_dict, block_enl, trerr, energy, Ψ, trmat, ee, es, Sj, newblock, newtensor_dict, newblock_enl, trmats)
    DMRGStepResult(block, tensor_dict, block_enl, trerr, energy, Ψ, trmat, ee, es, Sj)
end

function _dmrg_step_result(::Val{true}, block, tensor_dict, block_enl, trerr, energy, Ψ, trmat, ee, es, Sj, newblock, newtensor_dict, newblock_enl, trmats)
    DMRGStepResultWithEnv(block, tensor_dict, block_enl, trerr, energy, Ψ, trmat, ee, es, newblock[2], newtensor_dict[2], newblock_enl[2], trmats[2])
end

function dmrg_step_result!(args...; kwargs...)
    dmrg_step!(args...; kwargs...)
end
