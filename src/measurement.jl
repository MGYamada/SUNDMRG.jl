"""
tensor_dict, Sj = measurement!(SiSj, sys_label, sys, env, sys_enl, env_enl, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0))
measurement phase
"""
function measurement!(SiSj, sys_label, sys, env, sys_enl, env_enl, Ly, x_conn, y_conn, sys_connS, env_connS, sys_len, env_len, sys_tensor_dict, env_tensor_dict, sys_tensor_dict_hold, env_tensor_dict_hold, sys_αs, env_αs, sys_βs, env_βs, sys_dp, env_dp, sys_ms, env_ms, sys_enlarge, env_enlarge, superblock_H2, OM, Ψ0, transformation_matrix, comm, rank, Ncpu, engine; correlation = :none, margin = 0, lattice = :square, Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0))
    tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()

    for x in 1 : min(sys_enl.length, Ly)
        if rank == 0 && (lattice != :honeycombZC || x == x_conn || ((mod1(sys.length, 2Ly) <= Ly) ? x == 1 : x == Ly) || (x <= x_conn ? iseven(x) : isodd(x)))
            if x == x_conn
                if isnothing(sys_connS)
                    # if engine <: GPUEngine
                    #     Stemp = [@. CUSPARSE.CuSparseMatrixCSC(sys_enl.tensor_dict[x_conn][i, j]) for i in 1 : sys_len, j in 1 : sys_len]
                    # else
                    Stemp = sys_enl.tensor_dict[x_conn]
                    # end
                else
                    Stemp = sys_connS
                end
            else
                if haskey(sys_tensor_dict_hold, x)
                    Sx = sys_tensor_dict_hold[x]
                else
                    len = length(sys_αs)
                    if engine <: GPUEngine
                        Sx = [CuArray.(sys_tensor_dict[x][i, j]) for i in 1 : len, j in 1 : len]
                    else
                        Sx = sys_tensor_dict[x]
                    end
                end
                if engine <: GPUEngine
                    Stemp = [[CUDA.zeros(Float64, sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                else
                    Stemp = [[zeros(sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                end
                for e in sys_enlarge
                    @. Stemp[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sx[e.ki, e.kj][e.τ2]
                end
            end
            tensor_dict[x] = map(k -> isempty(Stemp[k...]) ? Matrix{Float64}[] : [Array(transformation_matrix[k[1]]' * (M * transformation_matrix[k[2]])) for M in Stemp[k...]], [(ki, kj) for ki in 1 : sys_len, kj in 1 : sys_len])
        end

        if correlation == :nn
            if x == x_conn
                if rank == 0
                    sys_S = Stemp
                end

                if sys_label == :l && sys.length == 0
                    bond = (1, 2)
                else
                    bond = (env_enl.length, env_enl.length + 1)
                end

                if rank == 0
                    if isnothing(env_connS)
                        # if engine <: GPUEngine
                        #     env_S = [@. CUSPARSE.CuSparseMatrixCSC(env_enl.tensor_dict[y_conn][i, j]) for i in 1 : env_len, j in 1 : env_len]
                        # else
                        env_S = env_enl.tensor_dict[y_conn]
                        # end
                    else
                        env_S = env_connS
                    end
                end

                SiSjΨ0 = deepcopy(Ψ0)

                if rank == 0
                    if engine <: GPUEngine
                        Ψtemp = [[CUDA.zeros(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                    else
                        Ψtemp = [[zeros(env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                    end
                end

                for s in superblock_H2
                    root1 = (s.sys_in + s.env_in - 2) % Ncpu

                    if rank == root1
                        temp3 = Ψ0[s.env_in, s.sys_in][s.om1]
                    else
                        if engine <: GPUEngine
                            temp3 = CuMatrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                        else
                            temp3 = Matrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                        end
                    end

                    if engine <: GPUEngine
                        CUDA.synchronize()
                    end
                    if root1 != 0
                        if rank == root1
                            MPI.Send(temp3, 0, root1, comm)
                        end
                        if rank == 0
                            MPI.Recv!(temp3, root1, root1, comm)
                        end
                    end

                    if rank == 0
                        for m in s.miniblock
                            temp4 = env_S[m.env_out, s.env_in][m.env_τ] * (sys_S[m.sys_out, s.sys_in][m.sys_τ] * temp3')'
                            @. Ψtemp[m.env_out, m.sys_out][m.om2] += m.coeff * temp4
                        end
                    end
                end

                for ki in 1 : env_len, kj in 1 : sys_len
                    for om in 1 : OM[kj, ki]
                        root2 = (kj + ki - 2) % Ncpu
                        if engine <: GPUEngine
                            CUDA.synchronize()
                        end
                        if root2 != 0
                            if rank == 0
                                MPI.Send(Ψtemp[ki, kj][om], root2, root2, comm)
                            end
                            if rank == root2
                                MPI.Recv!(SiSjΨ0[ki, kj][om], 0, root2, comm)
                            end
                        elseif rank == 0
                            SiSjΨ0[ki, kj][om] .= Ψtemp[ki, kj][om]
                        end
                    end
                end

                sisj = MPI.Reduce(dot(Ψ0, SiSjΨ0), MPI.SUM, 0, comm)

                if rank == 0
                    SiSj[bond] = sisj
                else
                    SiSj[bond] = 0.0
                end

            elseif env_enl.length % Ly == 0 && (lattice != :honeycombZC || (x_conn == 1 ? isodd(x) : iseven(x)))
                if rank == 0
                    sys_S = Stemp
                end

                if x_conn == 1
                    bond = (env_enl.length - x + 1, env_enl.length + x)
                else
                    bond = (env_enl.length - Ly + x, env_enl.length + 1 + Ly - x)
                end

                if rank == 0
                    if haskey(env_tensor_dict_hold, x)
                        Sx2 = env_tensor_dict_hold[x]
                    else
                        len = length(env_αs)
                        if engine <: GPUEngine
                            Sx2 = [CuArray.(env_tensor_dict[x][i, j]) for i in 1 : len, j in 1 : len]
                        else
                            Sx2 = env_tensor_dict[x]
                        end
                    end
                    if engine <: GPUEngine
                        env_S = [[CUDA.zeros(Float64, env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                    else
                        env_S = [[zeros(env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                    end
                    for e in env_enlarge
                        @. env_S[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sx2[e.ki, e.kj][e.τ2]
                    end
                end

                SiSjΨ0 = deepcopy(Ψ0)

                if rank == 0
                    if engine <: GPUEngine
                        Ψtemp = [[CUDA.zeros(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                    else
                        Ψtemp = [[zeros(env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                    end
                end

                for s in superblock_H2
                    root1 = (s.sys_in + s.env_in - 2) % Ncpu

                    if rank == root1
                        temp3 = Ψ0[s.env_in, s.sys_in][s.om1]
                    else
                        if engine <: GPUEngine
                            temp3 = CuMatrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                        else
                            temp3 = Matrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                        end
                    end

                    if engine <: GPUEngine
                        CUDA.synchronize()
                    end
                    if root1 != 0
                        if rank == root1
                            MPI.Send(temp3, 0, root1, comm)
                        end
                        if rank == 0
                            MPI.Recv!(temp3, root1, root1, comm)
                        end
                    end

                    if rank == 0
                        for m in s.miniblock
                            temp4 = env_S[m.env_out, s.env_in][m.env_τ] * (sys_S[m.sys_out, s.sys_in][m.sys_τ] * temp3')'
                            @. Ψtemp[m.env_out, m.sys_out][m.om2] += m.coeff * temp4
                        end
                    end
                end

                for ki in 1 : env_len, kj in 1 : sys_len
                    for om in 1 : OM[kj, ki]
                        root2 = (kj + ki - 2) % Ncpu
                        if engine <: GPUEngine
                            CUDA.synchronize()
                        end
                        if root2 != 0
                            if rank == 0
                                MPI.Send(Ψtemp[ki, kj][om], root2, root2, comm)
                            end
                            if rank == root2
                                MPI.Recv!(SiSjΨ0[ki, kj][om], 0, root2, comm)
                            end
                        elseif rank == 0
                            SiSjΨ0[ki, kj][om] .= Ψtemp[ki, kj][om]
                        end
                    end
                end

                sisj = MPI.Reduce(dot(Ψ0, SiSjΨ0), MPI.SUM, 0, comm)

                if rank == 0
                    SiSj[bond] = sisj
                else
                    SiSj[bond] = 0.0
                end
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
                    if rank == 0
                        if haskey(env_tensor_dict_hold, y)
                            Sy = env_tensor_dict_hold[y]
                        else
                            len = length(env_αs)
                            if engine <: GPUEngine
                                Sy = [CuArray.(env_tensor_dict[y][i, j]) for i in 1 : len, j in 1 : len]
                            else
                                Sy = env_tensor_dict[y]
                            end
                        end
                        if engine <: GPUEngine
                            env_S = [[CUDA.zeros(Float64, env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                        else
                            env_S = [[zeros(env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                        end
                        for e in env_enlarge
                            @. env_S[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sy[e.ki, e.kj][e.τ2]
                        end
                    end

                    SiSjΨ0 = deepcopy(Ψ0)

                    if rank == 0
                        if engine <: GPUEngine
                            Ψtemp = [[CUDA.zeros(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                        else
                            Ψtemp = [[zeros(env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                        end
                    end

                    for s in superblock_H2
                        root1 = (s.sys_in + s.env_in - 2) % Ncpu

                        if rank == root1
                            temp3 = Ψ0[s.env_in, s.sys_in][s.om1]
                        else
                            if engine <: GPUEngine
                                temp3 = CuMatrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                            else
                                temp3 = Matrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                            end
                        end

                        if engine <: GPUEngine
                            CUDA.synchronize()
                        end
                        if root1 != 0
                            if rank == root1
                                MPI.Send(temp3, 0, root1, comm)
                            end
                            if rank == 0
                                MPI.Recv!(temp3, root1, root1, comm)
                            end
                        end

                        if rank == 0
                            for m in s.miniblock
                                temp4 = env_S[m.env_out, s.env_in][m.env_τ] * (sys_S[m.sys_out, s.sys_in][m.sys_τ] * temp3')'
                                @. Ψtemp[m.env_out, m.sys_out][m.om2] += m.coeff * temp4
                            end
                        end
                    end

                    for ki in 1 : env_len, kj in 1 : sys_len
                        for om in 1 : OM[kj, ki]
                            root2 = (kj + ki - 2) % Ncpu
                            if engine <: GPUEngine
                                CUDA.synchronize()
                            end
                            if root2 != 0
                                if rank == 0
                                    MPI.Send(Ψtemp[ki, kj][om], root2, root2, comm)
                                end
                                if rank == root2
                                    MPI.Recv!(SiSjΨ0[ki, kj][om], 0, root2, comm)
                                end
                            elseif rank == 0
                                SiSjΨ0[ki, kj][om] .= Ψtemp[ki, kj][om]
                            end
                        end
                    end

                    sisj = MPI.Reduce(dot(Ψ0, SiSjΨ0), MPI.SUM, 0, comm)

                    if rank == 0
                        SiSj[bond] = sisj
                    else
                        SiSj[bond] = 0.0
                    end
                end
            end
        end
    end

    if correlation == :chain
        if rank == 0
            if !isempty(Sj) && (sys_label == :l && sys.length == 0)
                len = length(env_αs)
                if engine <: GPUEngine
                    Sj2 = [CuArray.(Sj[i, j]) for i in 1 : len, j in 1 : len]
                else
                    Sj2 = Sj
                end
                if engine <: GPUEngine
                    env_S = [[CUDA.zeros(Float64, env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                else
                    env_S = [[zeros(env_ms[i], env_ms[j]) for τ1 in 1 : get(env_dp[j], env_βs[i], 0)] for i in 1 : env_len, j in 1 : env_len]
                end
                for e in env_enlarge
                    @. env_S[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sj2[e.ki, e.kj][e.τ2]
                end
            elseif !isempty(Sj)
                len = length(sys_αs)
                if engine <: GPUEngine
                    Sj2 = [CuArray.(Sj[i, j]) for i in 1 : len, j in 1 : len]
                else
                    Sj2 = Sj
                end
                if engine <: GPUEngine
                    sys_S = [[CUDA.zeros(Float64, sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                else
                    sys_S = [[zeros(sys_ms[i], sys_ms[j]) for τ1 in 1 : get(sys_dp[j], sys_βs[i], 0)] for i in 1 : sys_len, j in 1 : sys_len]
                end
                for e in sys_enlarge
                    @. sys_S[e.i, e.j][e.τ1][e.range_i, e.range_j] += e.coeff * Sj2[e.ki, e.kj][e.τ2]
                end
            elseif x_conn == 1 && sys.length ÷ Ly == margin
                if isnothing(sys_connS)
                    # if engine <: GPUEngine
                    #     sys_S = [@. CUSPARSE.CuSparseMatrixCSC(sys_enl.tensor_dict[x_conn][i, j]) for i in 1 : sys_len, j in 1 : sys_len]
                    # else
                    sys_S = sys_enl.tensor_dict[x_conn]
                    # end
                else
                    sys_S = sys_connS
                end
            end
        end

        if !isempty(Sj) || (x_conn == 1 && sys.length ÷ Ly == margin)
            if sys_label == :l && sys.length == 0
                N = sys_enl.length + env_enl.length
                bond = (1, iseven(margin) ? N - margin * Ly : N - (margin + 1) * Ly + 1)

                if rank == 0
                    if isnothing(sys_connS)
                        # if engine <: GPUEngine
                        #     sys_S = [@. CUSPARSE.CuSparseMatrixCSC(sys_enl.tensor_dict[x_conn][i, j]) for i in 1 : sys_len, j in 1 : sys_len]
                        # else
                        sys_S = sys_enl.tensor_dict[x_conn]
                        # end
                    else
                        sys_S = sys_connS
                    end
                end
            elseif y_conn == 1 || (lattice == :honeycombZC && y_conn <= 2)
                N = sys_enl.length + env_enl.length
                bond = (env_enl.length, iseven(margin) ? N - margin * Ly : N - (margin + 1) * Ly + 1)

                if rank == 0
                    if isnothing(env_connS)
                        # if engine <: GPUEngine
                        #     env_S = [@. CUSPARSE.CuSparseMatrixCSC(env_enl.tensor_dict[y_conn][i, j]) for i in 1 : env_len, j in 1 : env_len]
                        # else
                        env_S = env_enl.tensor_dict[y_conn]
                        # end
                    else
                        env_S = env_connS
                    end
                end
            end

            if (sys_label == :l && sys.length == 0) || y_conn == 1 || (lattice == :honeycombZC && y_conn <= 2)
                SiSjΨ0 = deepcopy(Ψ0)

                if rank == 0
                    if engine <: GPUEngine
                        Ψtemp = [[CUDA.zeros(Float64, env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                    else
                        Ψtemp = [[zeros(env_ms[ki], sys_ms[kj]) for J in 1 : OM[kj, ki]] for ki in 1 : env_len, kj in 1 : sys_len]
                    end
                end

                for s in superblock_H2
                    root1 = (s.sys_in + s.env_in - 2) % Ncpu

                    if rank == root1
                        temp3 = Ψ0[s.env_in, s.sys_in][s.om1]
                    else
                        if engine <: GPUEngine
                            temp3 = CuMatrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                        else
                            temp3 = Matrix{Float64}(undef, s.env_in_size, s.sys_in_size)
                        end
                    end

                    if engine <: GPUEngine
                        CUDA.synchronize()
                    end
                    if root1 != 0
                        if rank == root1
                            MPI.Send(temp3, 0, root1, comm)
                        end
                        if rank == 0
                            MPI.Recv!(temp3, root1, root1, comm)
                        end
                    end

                    if rank == 0
                        for m in s.miniblock
                            temp4 = env_S[m.env_out, s.env_in][m.env_τ] * (sys_S[m.sys_out, s.sys_in][m.sys_τ] * temp3')'
                            @. Ψtemp[m.env_out, m.sys_out][m.om2] += m.coeff * temp4
                        end
                    end
                end

                for ki in 1 : env_len, kj in 1 : sys_len
                    for om in 1 : OM[kj, ki]
                        root2 = (kj + ki - 2) % Ncpu
                        if engine <: GPUEngine
                            CUDA.synchronize()
                        end
                        if root2 != 0
                            if rank == 0
                                MPI.Send(Ψtemp[ki, kj][om], root2, root2, comm)
                            end
                            if rank == root2
                                MPI.Recv!(SiSjΨ0[ki, kj][om], 0, root2, comm)
                            end
                        elseif rank == 0
                            SiSjΨ0[ki, kj][om] .= Ψtemp[ki, kj][om]
                        end
                    end
                end

                sisj = MPI.Reduce(dot(Ψ0, SiSjΨ0), MPI.SUM, 0, comm)

                if rank == 0
                    SiSj[bond] = sisj
                else
                    SiSj[bond] = 0.0
                end
            end

            if rank == 0
                Sj::Matrix{Vector{Matrix{Float64}}} = map(k -> isempty(sys_S[k...]) ? Matrix{Float64}[] : [Array(transformation_matrix[k[1]]' * (M * transformation_matrix[k[2]])) for M in sys_S[k...]], [(ki, kj) for ki in 1 : sys_len, kj in 1 : sys_len])
            else
                Sj = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
            end
        end
    end

    tensor_dict, Sj
end