"""
dc = dcinit(N, Ncpu)
Divide conquer
"""
function dcinit(N, Ncpu)
    @. N * [0 : Ncpu;] ÷ Ncpu
end

"""
Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
Function barrier
"""
function Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
    Ψtemp = [[zeros_like_engine(engine, Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]

    for s in superblock_H2
        root1 = (s.sys_in + s.env_in - 2) % Ncpu
        if rank == root1
            temp3 = Ψin[s.env_in, s.sys_in][s.om1]
        else
            temp3 = engine_matrix_type(engine)(undef, s.env_in_size, s.sys_in_size)
        end
        synchronize_engine(engine)
        MPI.Bcast!(temp3, root1, comm)

        if (x, y) != (0, 0)
            for m in s.miniblock
                temp4 = env_S[m.env_out, s.env_in][m.env_τ] * (sys_S[m.sys_out, s.sys_in][m.sys_τ] * temp3')'
                @. Ψtemp[m.env_out, m.sys_out][m.om2] += m.coeff * temp4
            end
        end
    end

    for ki in 1 : env_len, kj in 1 : sys_len
        for om in 1 : OM[kj, ki]
            root2 = (kj + ki - 2) % Ncpu
            synchronize_engine(engine)
            MPI.Reduce!(Ψtemp[ki, kj][om], MPI.SUM, root2, comm)
            if rank == root2
                Ψout[ki, kj][om] .+= Ψtemp[ki, kj][om]
            end
        end
    end
end
function _step_spin_operator(x, conn, connS, ms, βs, dp, enlarging, tensor_dict_hold, engine)
    if x == conn
        return connS
    elseif x > 0
        return _enlarged_spin_tensor(ms, βs, dp, enlarging, tensor_dict_hold[x], engine)
    end

    return nothing
end

function _run_step_lanczos!(Ψ0, context::_StepLanczosContext)
    (; target, comm, rank, engine, alg, superblock_H1, bonds_hold, x_conn, y_conn, sys_connS, env_connS, sys_ms, env_ms, sys_βs, env_βs, sys_dp, env_dp, sys_enlarge, env_enlarge, sys_tensor_dict_hold, env_tensor_dict_hold, superblock_H2, OM, sys_len, env_len, Ncpu) = context

    local E
    time_Lanczos = @elapsed E = Lanczos!(Ψ0, target + 1, comm, rank, engine; alg = alg) do Ψout, Ψin
        for s in superblock_H1
            for J in eachindex(Ψin[s.env_ind, s.sys_ind])
                temp1 = Ψin[s.env_ind, s.sys_ind][J] * s.sys_H'
                temp2 = s.env_H * Ψin[s.env_ind, s.sys_ind][J]
                @. Ψout[s.env_ind, s.sys_ind][J] = temp1 + temp2
            end
        end

        for (x, y) in bonds_hold
            sys_S = _step_spin_operator(x, x_conn, sys_connS, sys_ms, sys_βs, sys_dp, sys_enlarge, sys_tensor_dict_hold, engine)
            env_S = _step_spin_operator(y, y_conn, env_connS, env_ms, env_βs, env_dp, env_enlarge, env_tensor_dict_hold, engine)
            Lanczos_kernel!(Ψout, Ψin, x, y, sys_S, env_S, superblock_H2, OM, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
        end
    end

    return E, time_Lanczos
end

function _step_energy(E, time_Lanczos, sys_enl, env_enl, superblock_bonds, comm, rank, noisy, ::Val{Nc}) where Nc
    time1 = MPI.Reduce(time_Lanczos, MPI.MAX, 0, comm)
    if rank == 0
        if noisy
            println(time1, " seconds elapsed in the Lanczos method")
        end

        if Nc == 2
            return 0.5E
        end

        Nbond = length(sys_enl.bonds) + length(env_enl.bonds) + length(superblock_bonds)
        return E + Nbond / Nc
    end

    return 0.0
end

_report_overlap(Ψ0_guess::Nothing, Ψ0, comm, rank, noisy) = nothing

function _report_overlap(Ψ0_guess, Ψ0, comm, rank, noisy)
    nume = MPI.Reduce(mydot(Ψ0_guess, Ψ0), MPI.SUM, 0, comm)
    den1 = MPI.Reduce(mydot(Ψ0_guess, Ψ0_guess), MPI.SUM, 0, comm)
    den2 = MPI.Reduce(mydot(Ψ0, Ψ0), MPI.SUM, 0, comm)
    if noisy && rank == 0
        println("overlap |<ψ_guess|ψ>| = ", abs(nume) / sqrt(den1 * den2))
    end
    return nothing
end
