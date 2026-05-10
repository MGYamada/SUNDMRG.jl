_connection_or_tensor(connS, block_enl, conn) = isnothing(connS) ? block_enl.tensor_dict[conn] : connS
_held_or_stored_tensor(tensor_dict_hold, tensor_dict, site) = haskey(tensor_dict_hold, site) ? tensor_dict_hold[site] : tensor_dict[site]

function _enlarged_spin_tensor(ms, βs, dp, enlarging, source, engine)
    spin = [[zeros_like_engine(engine, Float64, ms[i], ms[j]) for _ in 1 : get(dp[j], βs[i], 0)] for i in eachindex(ms), j in eachindex(ms)]
    for e in enlarging
        @. spin[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * source[e.ki, e.kj][e.τ2]
    end
    return spin
end

function _project_spin_tensor(spin, transformation_matrix, len, engine)
    map([(ki, kj) for ki in 1 : len, kj in 1 : len]) do k
        isempty(spin[k...]) && return empty_engine_matrix_vector(engine)
        [transformation_matrix[k[1]]' * (M * transformation_matrix[k[2]]) for M in spin[k...]]
    end
end

function _host_project_spin_tensor(spin, transformation_matrix, len)
    map([(ki, kj) for ki in 1 : len, kj in 1 : len]) do k
        isempty(spin[k...]) && return Matrix{Float64}[]
        [Array(transformation_matrix[k[1]]' * (M * transformation_matrix[k[2]])) for M in spin[k...]]
    end
end

function _store_sisj_bond!(SiSj, bond, sys_S, env_S, superblock_H2, OM, Ψ0, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
    SiSjΨ0 = deepcopy(Ψ0)

    if isroot(rank)
        Ψtemp = [[zeros_like_engine(engine, Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
    end

    for s in superblock_H2
        root1 = (s.sys_in + s.env_in - 2) % Ncpu

        if rank == root1
            temp3 = Ψ0[s.env_in, s.sys_in][s.om1]
        else
            temp3 = engine_matrix_type(engine)(undef, s.env_in_size, s.sys_in_size)
        end

        synchronize_engine(engine)
        if root1 != 0
            if rank == root1
                MPI.Send(temp3, 0, root1, comm)
            end
            if isroot(rank)
                MPI.Recv!(temp3, root1, root1, comm)
            end
        end

        if isroot(rank)
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
            if root2 != 0
                if isroot(rank)
                    MPI.Send(Ψtemp[ki, kj][om], root2, root2, comm)
                end
                if rank == root2
                    MPI.Recv!(SiSjΨ0[ki, kj][om], 0, root2, comm)
                end
            elseif isroot(rank)
                SiSjΨ0[ki, kj][om] .= Ψtemp[ki, kj][om]
            end
        end
    end

    sisj = MPI.Reduce(mydot(Ψ0, SiSjΨ0), MPI.SUM, 0, comm)

    if isroot(rank)
        SiSj[bond] = sisj
    else
        SiSj[bond] = 0.0
    end

    SiSj
end

"""
tensor_dict, Sj = measurement!(SiSj, sys_label, sys, env, sys_enl, env_enl, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0))
measurement phase
"""
function measurement!(SiSj, sys_label, sys, env, sys_enl, env_enl, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0))
    tensor_dict = empty_engine_tensor_dict(engine)
    sys_S = nothing
    env_S = nothing

    for x in 1 : min(sys_enl.length, Ly)
        if isroot(rank) && (lattice != :honeycombZC || x == x_conn || ((mod1(sys.length, 2Ly) <= Ly) ? x == 1 : x == Ly) || (x <= x_conn ? iseven(x) : isodd(x)))
            if x == x_conn
                Stemp = _connection_or_tensor(sys_connS, sys_enl, x_conn)
            else
                Sx = _held_or_stored_tensor(sys_tensor_dict_hold, sys_tensor_dict, x)
                Stemp = _enlarged_spin_tensor(sys_ms, sys_βs, sys_dp, sys_enlarge, Sx, engine)
            end
            tensor_dict[x] = _project_spin_tensor(Stemp, transformation_matrix, sys_len, engine)
        end

        if correlation == :nn
            if x == x_conn
                if isroot(rank)
                    sys_S = Stemp
                end

                if sys_label == :l && sys.length == 0
                    bond = (1, 2)
                else
                    bond = (env_enl.length, env_enl.length + 1)
                end

                if isroot(rank)
                    env_S = _connection_or_tensor(env_connS, env_enl, y_conn)
                end

                _store_sisj_bond!(SiSj, bond, sys_S, env_S, superblock_H2, OM, Ψ0, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)

            elseif env_enl.length % Ly == 0 && (lattice != :honeycombZC || (x_conn == 1 ? isodd(x) : iseven(x)))
                if isroot(rank)
                    sys_S = Stemp
                end

                if x_conn == 1
                    bond = (env_enl.length - x + 1, env_enl.length + x)
                else
                    bond = (env_enl.length - Ly + x, env_enl.length + 1 + Ly - x)
                end

                if isroot(rank)
                    Sx2 = _held_or_stored_tensor(env_tensor_dict_hold, env_tensor_dict, x)
                    env_S = _enlarged_spin_tensor(env_ms, env_βs, env_dp, env_enlarge, Sx2, engine)
                end

                _store_sisj_bond!(SiSj, bond, sys_S, env_S, superblock_H2, OM, Ψ0, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
            end

            if Ly > 2 && sys_label == :r && env_enl.length % Ly != 0
                if mod1(sys_enl.length, 2Ly) <= Ly
                    x2, y = 1, Ly
                else
                    x2, y = Ly, 1
                end
                q = (env_enl.length ÷ Ly) * Ly
                bond = (q + 1, q + Ly)
                if x == x2 && !haskey(SiSj, bond)
                    if isroot(rank)
                        Sy = _held_or_stored_tensor(env_tensor_dict_hold, env_tensor_dict, y)
                        env_S = _enlarged_spin_tensor(env_ms, env_βs, env_dp, env_enlarge, Sy, engine)
                    end

                    _store_sisj_bond!(SiSj, bond, sys_S, env_S, superblock_H2, OM, Ψ0, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
                end
            end
        end
    end

    if correlation == :chain
        if isroot(rank)
            if !isempty(Sj) && (sys_label == :l && sys.length == 0)
                Sj2 = engine <: GPUEngine ? [to_engine_array.(Ref(engine), Sj[i, j]) for i in 1 : length(env_αs), j in 1 : length(env_αs)] : Sj
                env_S = _enlarged_spin_tensor(env_ms, env_βs, env_dp, env_enlarge, Sj2, engine)
            elseif !isempty(Sj)
                Sj2 = engine <: GPUEngine ? [to_engine_array.(Ref(engine), Sj[i, j]) for i in 1 : length(sys_αs), j in 1 : length(sys_αs)] : Sj
                sys_S = _enlarged_spin_tensor(sys_ms, sys_βs, sys_dp, sys_enlarge, Sj2, engine)
            elseif x_conn == 1 && sys.length ÷ Ly == margin
                sys_S = _connection_or_tensor(sys_connS, sys_enl, x_conn)
            end
        end

        if !isempty(Sj) || (x_conn == 1 && sys.length ÷ Ly == margin)
            if sys_label == :l && sys.length == 0
                N = sys_enl.length + env_enl.length
                bond = (1, iseven(margin) ? N - margin * Ly : N - (margin + 1) * Ly + 1)

                if isroot(rank)
                    sys_S = _connection_or_tensor(sys_connS, sys_enl, x_conn)
                end
            elseif y_conn == 1 || (lattice == :honeycombZC && y_conn <= 2)
                N = sys_enl.length + env_enl.length
                bond = (env_enl.length, iseven(margin) ? N - margin * Ly : N - (margin + 1) * Ly + 1)

                if isroot(rank)
                    env_S = _connection_or_tensor(env_connS, env_enl, y_conn)
                end
            end

            if (sys_label == :l && sys.length == 0) || y_conn == 1 || (lattice == :honeycombZC && y_conn <= 2)
                _store_sisj_bond!(SiSj, bond, sys_S, env_S, superblock_H2, OM, Ψ0, sys_len, env_len, sys_ms, env_ms, comm, rank, Ncpu, engine)
            end

            if isroot(rank)
                Sj::Matrix{Vector{Matrix{Float64}}} = _host_project_spin_tensor(sys_S, transformation_matrix, sys_len)
            else
                Sj = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
            end
        end
    end

    tensor_dict, Sj
end
