struct DMRGOutput{Nc}
    ms::Vector{Tuple{Int, Float64}}
    errors::Vector{Float64}
    energies::Vector{Float64}
    EEs::Vector{Float64}
    EE::Vector{Float64}
    ES::Dict{NTuple{Nc, Int}, Vector{Float64}}
    SiSj::Dict{Tuple{Int, Int}, Float64}
end

abstract type Model end
struct HeisenbergModel <: Model end # No parameter

abstract type Symmetry end
struct SU{Nc} <: Symmetry end
SU(Nc) = SU{Nc}()

abstract type SymmetricModel{S, M} <: Model where {S <: Symmetry, M <: Model} end

struct HeisenbergModelSU{Nc} <: SymmetricModel{SU{Nc}, HeisenbergModel} end

Base.:*(::SU{Nc}, ::HeisenbergModel) where Nc = HeisenbergModelSU{Nc}()

abstract type Lattice{D} end

struct SquareLattice <: Lattice{2}
    Lx::Int
    Ly::Int
end

struct HoneycombLattice <: Lattice{2}
    Lx::Int
    Ly::Int
    BC::Symbol
    function HoneycombLattice(Lx, Ly, BC)
        @assert iseven(Ly)
        @assert BC == :ZC
        new(Lx, Ly, BC)
    end
end

abstract type Engine end
abstract type CPUEngine <: Engine end
abstract type GPUEngine <: Engine end

"""
string = graphic(sys_block, env_block; sys_label = :l)
visualizes DMRG
"""
function graphic(sys_block, env_block; sys_label = :l)
    str = repeat("=", sys_block.length) * "**" * repeat("-", env_block.length)
    if sys_label == :r
        str = reverse(str)
    elseif sys_label != :l
        throw(ArgumentError("sys_label must be :l or :r"))
    end
    str
end

"""
    rank, dmrg = run_DMRG(model::HeisenbergModelSU{Nc}, lattice::Lattice{D}, m_warmup::Int, m_sweep_list::Vector{Int}, m_cooldown:Int, engine::Type{<:SUNDMRG.Engine}; target = 0, widthmax = 0, tables = nothing, fileio = false, scratch = ".", ES_max = 20.0, tol_energy = 1e-5, tol_EE = 1e-3, correlation = :none, margin = 0, alg = :slow)
    rank, dmrg = run_DMRG(model::HeisenbergModelSU{Nc}, lattice::Lattice{D}, m_warmup::Tuple{Int, Float64}, m_sweep_list::Vector{Tuple{Int, Float64}}, m_cooldown:Tuple{Int, Float64}, engine::Type{<:SUNDMRG.Engine}; target = 0, widthmax = 0, tables = nothing, fileio = false, scratch = ".", ES_max = 20.0, tol_energy = 1e-5, tol_EE = 1e-3, correlation = :none, margin = 0, alg = :slow)
doing the finite-system algorithm
(target = 0: ground state, target = 1: 1st excited state...)
Currently only suupports a small number for target
"""
function run_DMRG end

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::SquareLattice, m_warmup::Int, m_sweep_list::Vector{Int}, m_cooldown::Int, engine::Type{<:Engine}; kwargs...) where Nc
    _run_DMRG(model, :square, lat.Lx, lat.Ly, (m_warmup, 0.0), map(x -> (x, 0.0), m_sweep_list), (m_cooldown, 0.0), engine; kwargs...)
end

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::HoneycombLattice, m_warmup::Int, m_sweep_list::Vector{Int}, m_cooldown::Int, engine::Type{<:Engine}; kwargs...) where Nc
    if lat.BC == :ZC
        _run_DMRG(model, :honeycombZC, lat.Lx, lat.Ly, (m_warmup, 0.0), map(x -> (x, 0.0), m_sweep_list), (m_cooldown, 0.0), engine; kwargs...)
    end
end

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::SquareLattice, m_warmup::Tuple{Int, Float64}, m_sweep_list::Vector{Tuple{Int, Float64}}, m_cooldown::Tuple{Int, Float64}, engine::Type{<:Engine}; kwargs...) where Nc
    _run_DMRG(model, :square, lat.Lx, lat.Ly, m_warmup, m_sweep_list, m_cooldown, engine; kwargs...)
end

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::HoneycombLattice, m_warmup::Tuple{Int, Float64}, m_sweep_list::Vector{Tuple{Int, Float64}}, m_cooldown::Tuple{Int, Float64}, engine::Type{<:Engine}; kwargs...) where Nc
    if lat.BC == :ZC
        _run_DMRG(model, :honeycombZC, lat.Lx, lat.Ly, m_warmup, m_sweep_list, m_cooldown, engine; kwargs...)
    end
end

function _run_DMRG(model::HeisenbergModelSU{Nc}, lattice, Lx, Ly, m_warmup, m_sweep_list, m_cooldown, engine; target = 0, widthmax = 0, tables = nothing, fileio = false, scratch = ".", ES_max = 20.0, tol_energy = 1e-5, tol_EE = 1e-3, correlation = :none, margin = 0, alg = :slow) where Nc
    @assert Nc >= 2

    on_the_fly = Nc == 2
    mirror = lattice == :square || lattice == :honeycombZC

    @assert iseven(Lx)
    @assert lattice == :square || lattice == :honeycombZC
    @assert correlation == :none || correlation == :nn || correlation == :chain
    @assert alg == :slow || alg == :fast

    if on_the_fly
        @assert (Lx * Ly) % Nc == 0
    else
        if iseven(Nc)
            @assert Ly % (Nc >> 1) == 0
        else
            @assert Ly % Nc == 0
        end
    end

    γ_list = SUNIrrep{Nc}[]
    for h in ((1 : Nc) .% Nc)
        push!(γ_list, SUNIrrep(ntuple(i -> 0 + (i <= h), Val(Nc))))
    end

    MPI.Init_thread(MPI.THREAD_FUNNELED)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Ncpu = MPI.Comm_size(comm)

    if engine <: GPUEngine
        Ngpu = Int(length(devices()))
        @assert Ncpu <= Ngpu
        device!(rank)
        magma_init()
    end

    N = Lx * Ly
    signfactor = iseven(Nc) ? -1.0 : 1.0
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

    if rank == 0
        println(repeat("-", 60))
        println("SU($Nc) DMRG simulation on the $Lx x $Ly $lattice lattice cylinder:")
        if on_the_fly
            println("All representations are used in the calculation.")
        else
            irreps = irreplist(Nc, widthmax)
            println(length(irreps), " irreps from ", weight(first(irreps)), " to ", weight(last(irreps)), " are used in the calculation.")
        end
        println(target == 0 ? "The ground state" : "The excited state #$target", " will be calculated.")
        println(repeat("-", 60))

        if fileio
            dirid = lpad(rand(0 : 99999), 5, "0")
            mkdir("$scratch/temp$dirid")
        else
            dirid = 0
        end

        blockL = Block(0, Tuple{Int, Int}[], [trivialirrep(Val(Nc))], [1], [1], Dict{Symbol, Vector{Matrix{Float64}}}(:H => [zeros(1, 1)]))
        blockL_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
        if engine <: GPUEngine
            trmatL = [CuArray(diagm([1.0]))]
        else
            trmatL = [diagm([1.0])]
        end

        if !mirror
            blockR = Block(0, Tuple{Int, Int}[], [trivialirrep(Val(Nc))], [1], [1], Dict{Symbol, Vector{Matrix{Float64}}}(:H => [zeros(1, 1)]))
            blockR_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
            if engine <: GPUEngine
                trmatR = [CuArray(diagm([1.0]))]
            else
                trmatR = [diagm([1.0])]
            end
        end

        if mirror
            if fileio
                jldsave("$scratch/temp$dirid/block_l_$(blockL.length).jld2"; env_block = blockL)
                jldsave("$scratch/temp$dirid/block_r_$(blockL.length).jld2"; env_block = blockL)
                jldsave("$scratch/temp$dirid/trmat_l_$(blockL.length).jld2"; env_trmat = Array.(trmatL))
                jldsave("$scratch/temp$dirid/trmat_r_$(blockL.length).jld2"; env_trmat = Array.(trmatL))
            else
                block_table[:l, blockL.length] = blockL
                block_table[:r, blockL.length] = blockL
                trmat_table[:l, blockL.length] = Array.(trmatL)
                trmat_table[:r, blockL.length] = Array.(trmatL)
            end
        else
            if fileio
                jldsave("$scratch/temp$dirid/block_l_$(blockL.length).jld2"; env_block = blockL)
                jldsave("$scratch/temp$dirid/trmat_l_$(blockL.length).jld2"; env_trmat = Array.(trmatL))
                jldsave("$scratch/temp$dirid/block_r_$(blockR.length).jld2"; env_block = blockR)
                jldsave("$scratch/temp$dirid/trmat_r_$(blockR.length).jld2"; env_trmat = Array.(trmatR))
            else
                block_table[:l, blockL.length] = blockL
                trmat_table[:l, blockL.length] = Array.(trmatL)
                block_table[:r, blockR.length] = blockR
                trmat_table[:r, blockR.length] = Array.(trmatR)
            end
        end

        println("#")
        println("# Warming up with (m, α) = ", m_warmup)
        println("#")
    else
        dirid = 0
        blockL = Block(0, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
        blockL_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()

        if !mirror
            blockR = Block(0, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
            blockR_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
        end
    end

    blockL_enl = enlarge_block(blockL, blockL_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    if !mirror
        blockR_enl = enlarge_block(blockR, blockR_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
    end

    if engine <: GPUEngine
        if mirror
            Ψ = [Matrix{Vector{CuMatrix{Float64}}}(undef, 0, 0)]
        else
            Ψ = [Matrix{Vector{CuMatrix{Float64}}}(undef, 0, 0), Matrix{Vector{CuMatrix{Float64}}}(undef, 0, 0)]
        end
    else
        if mirror
            Ψ = [Matrix{Vector{Matrix{Float64}}}(undef, 0, 0)]
        else
            Ψ = [Matrix{Vector{Matrix{Float64}}}(undef, 0, 0), Matrix{Vector{Matrix{Float64}}}(undef, 0, 0)]
        end
    end

    while blockL.length < Ly
        if rank == 0
            if mirror
                println(graphic(blockL, blockL))
            else
                println(graphic(blockL, blockR))
            end
        end

        if mirror
            blockL, blockL_tensor_dict, blockL_enl, trerr, energy, Ψ[1], trmatL, ee, es, = dmrg_step!(SiSj, :l, blockL, blockL, blockL_tensor_dict, blockL_tensor_dict, blockL_enl, blockL_enl, Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); lattice = lattice, alg = alg)
        else
            blockL, blockL_tensor_dict, blockL_enl, trerr, energy, Ψ[1], trmatL, ee, es, blockR, blockR_tensor_dict, blockR_enl, trmatR = dmrg_step!(SiSj, :l, blockL, blockR, blockL_tensor_dict, blockR_tensor_dict, blockL_enl, blockR_enl, Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(true); lattice = lattice, alg = alg)
            Ψ[2] = wavefunction_reverse(Ψ[1], :l, blockL, blockR, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
        end

        if 2blockL.length == N
            push!(m_list, m_warmup)
            push!(errors, trerr)
            push!(energies, energy)
            push!(EEs, ee)
            ES = es
        end

        if rank == 0
            println("E / N = ", energy / 2blockL.length)
            println("E     = ", energy)
            println("S_EE  = ", ee)
            if mirror
                if fileio
                    jldsave("$scratch/temp$dirid/block_l_$(blockL.length).jld2"; env_block = blockL)
                    jldsave("$scratch/temp$dirid/block_r_$(blockL.length).jld2"; env_block = blockL)
                    jldsave("$scratch/temp$dirid/trmat_l_$(blockL.length).jld2"; env_trmat = Array.(trmatL))
                    jldsave("$scratch/temp$dirid/trmat_r_$(blockL.length).jld2"; env_trmat = Array.(trmatL))
                else
                    block_table[:l, blockL.length] = blockL
                    block_table[:r, blockL.length] = blockL
                    trmat_table[:l, blockL.length] = Array.(trmatL)
                    trmat_table[:r, blockL.length] = Array.(trmatL)
                end
            else
                if fileio
                    jldsave("$scratch/temp$dirid/block_l_$(blockL.length).jld2"; env_block = blockL)
                    jldsave("$scratch/temp$dirid/trmat_l_$(blockL.length).jld2"; env_trmat = Array.(trmatL))
                    jldsave("$scratch/temp$dirid/block_r_$(blockR.length).jld2"; env_block = blockR)
                    jldsave("$scratch/temp$dirid/trmat_r_$(blockR.length).jld2"; env_trmat = Array.(trmatR))
                else
                    block_table[:l, blockL.length] = blockL
                    trmat_table[:l, blockL.length] = Array.(trmatL)
                    block_table[:r, blockR.length] = blockR
                    trmat_table[:r, blockR.length] = Array.(trmatR)
                end
            end
        end
    end

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
            if rank == 0
                if mirror
                    println(graphic(sys_blocks[1], sys_blocks[1]; sys_label = :l))
                else
                    println(graphic(sys_blocks[1], sys_blocks[2]; sys_label = :l))
                end
            end

            L = 2sys_block_enls[1].length
            if mirror
                sys_blocks[1], sys_tensor_dicts[1], sys_block_enls[1], trerr, energy, Ψ[1], sys_trmats[1], ee, es, = dmrg_step!(SiSj, :l, sys_blocks[1], sys_blocks[1], sys_tensor_dicts[1], sys_tensor_dicts[1], sys_block_enls[1], sys_block_enls[1], Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); lattice = lattice, alg = alg)
            else
                sys_blocks[1], sys_tensor_dicts[1], sys_block_enls[1], trerr, energy, Ψ[1], sys_trmats[1], ee, es, sys_blocks[2], sys_tensor_dicts[2], sys_block_enls[2], sys_trmats[2] = dmrg_step!(SiSj, :l, sys_blocks[1], sys_blocks[2], sys_tensor_dicts[1], sys_tensor_dicts[2], sys_block_enls[1], sys_block_enls[2], Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(true); lattice = lattice, alg = alg)
                Ψ[2] = wavefunction_reverse(Ψ[1], :l, sys_blocks[1], sys_blocks[2], widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)
            end

            if rank == 0
                println("E / N = ", energy / L)
                println("E     = ", energy)
                println("S_EE  = ", ee)
            end
        else
            for i in 1 : (mirror ? 1 : 2)
                if i == 1
                    sys_label, env_label = :l, :r
                else
                    sys_label, env_label = :r, :l
                end
                if sys_blocks[i].length % Ly == 0
                    if rank == 0
                        if fileio
                            env_block = load_object("$scratch/temp$dirid/block_$(env_label)_$(L - sys_blocks[i].length - 1).jld2")::Block{Nc}
                            if engine <: GPUEngine
                                env_trmats[i] = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_blocks[i].length - 1).jld2")::Vector{Matrix{Float64}})
                            else
                                env_trmats[i] = load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_blocks[i].length - 1).jld2")::Vector{Matrix{Float64}}
                            end
                        else
                            env_block = block_table[env_label, L - sys_blocks[i].length - 1]
                            if engine <: GPUEngine
                                env_trmats[i] = CuArray.(trmat_table[env_label, L - sys_blocks[i].length - 1])
                            else
                                env_trmats[i] = trmat_table[env_label, L - sys_blocks[i].length - 1]
                            end
                        end

                        env_tensor_dict = spin_operators!(tensor_table, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, fileio, scratch, dirid, block_table, trmat_table, on_the_fly, engine; lattice = lattice)
                    else
                        env_block = Block(L - sys_blocks[i].length - 1, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
                        env_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
                        env_trmats[i] = Matrix{Float64}[]
                    end
                
                    env_block_enls[i] = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)
                end

                Ψ0_guess = eig_prediction(Ψ[i], sys_label, sys_block_enls[i], env_block_enls[i], sys_trmats[i], env_trmats[i], widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)

                if rank == 0
                    if fileio
                        env_block = load_object("$scratch/temp$dirid/block_$(env_label)_$(L - sys_blocks[i].length - 2).jld2")::Block{Nc}
                        if engine <: GPUEngine
                            env_trmats[i] = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_blocks[i].length - 2).jld2")::Vector{Matrix{Float64}})
                        else
                            env_trmats[i] = load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_blocks[i].length - 2).jld2")::Vector{Matrix{Float64}}
                        end
                    else
                        env_block = block_table[env_label, L - sys_blocks[i].length - 2]
                        if engine <: GPUEngine
                            env_trmats[i] = CuArray.(trmat_table[env_label, L - sys_blocks[i].length - 2])
                        else
                            env_trmats[i] = trmat_table[env_label, L - sys_blocks[i].length - 2]
                        end
                    end

                    env_tensor_dict = spin_operators!(tensor_table, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, fileio, scratch, dirid, block_table, trmat_table, on_the_fly, engine; lattice = lattice)
                else
                    env_block = Block(L - sys_blocks[i].length - 2, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
                    env_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
                    env_trmats[i] = Matrix{Float64}[]
                end

                env_block_enls[i] = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)

                if i == 1 && rank == 0
                    println(graphic(sys_blocks[i], env_block; sys_label = sys_label))
                end

                sys_blocks[i], sys_tensor_dicts[i], sys_block_enls[i], trerr, energy, Ψ[i], sys_trmats[i], ee, es, = dmrg_step!(SiSj, sys_label, sys_blocks[i], env_block, sys_tensor_dicts[i], env_tensor_dict, sys_block_enls[i], env_block_enls[i], Ly, m_warmup..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); Ψ0_guess = Ψ0_guess, lattice = lattice, alg = alg, noisy = i == 1)

                if i == 1 && rank == 0
                    println("E / N = ", energy / L)
                    println("E     = ", energy)
                    println("S_EE  = ", ee)
                end
            end
        end

        if L == N
            push!(m_list, m_warmup)
            push!(errors, trerr)
            push!(energies, energy)
            push!(EEs, ee)
            ES = es
        end

        if rank == 0
            if mirror
                if fileio
                    jldsave("$scratch/temp$dirid/block_l_$(sys_blocks[1].length).jld2"; env_block = sys_blocks[1])
                    jldsave("$scratch/temp$dirid/block_r_$(sys_blocks[1].length).jld2"; env_block = sys_blocks[1])
                    jldsave("$scratch/temp$dirid/trmat_l_$(sys_blocks[1].length).jld2"; env_trmat = Array.(sys_trmats[1]))
                    jldsave("$scratch/temp$dirid/trmat_r_$(sys_blocks[1].length).jld2"; env_trmat = Array.(sys_trmats[1]))
                else
                    block_table[:l, sys_blocks[1].length] = sys_blocks[1]
                    block_table[:r, sys_blocks[1].length] = sys_blocks[1]
                    trmat_table[:l, sys_blocks[1].length] = Array.(sys_trmats[1])
                    trmat_table[:r, sys_blocks[1].length] = Array.(sys_trmats[1])
                end
            else
                if fileio
                    jldsave("$scratch/temp$dirid/block_l_$(sys_blocks[1].length).jld2"; env_block = sys_blocks[1])
                    jldsave("$scratch/temp$dirid/trmat_l_$(sys_blocks[1].length).jld2"; env_trmat = Array.(sys_trmats[1]))
                    jldsave("$scratch/temp$dirid/block_r_$(sys_blocks[2].length).jld2"; env_block = sys_blocks[2])
                    jldsave("$scratch/temp$dirid/trmat_r_$(sys_blocks[2].length).jld2"; env_trmat = Array.(sys_trmats[2]))
                else
                    block_table[:l, sys_blocks[1].length] = sys_blocks[1]
                    trmat_table[:l, sys_blocks[1].length] = Array.(sys_trmats[1])
                    block_table[:r, sys_blocks[2].length] = sys_blocks[2]
                    trmat_table[:r, sys_blocks[2].length] = Array.(sys_trmats[2])
                end
            end
        end
    end

    Ψ0 = Ψ[1]

    sys_label, env_label = :l, :r
    sys_block = sys_blocks[1]
    sys_tensor_dict = sys_tensor_dicts[1]
    sys_trmat = sys_trmats[1]
    sys_block_enl = sys_block_enls[1]

    if rank == 0
        if fileio
            env_block = load_object("$scratch/temp$dirid/block_$(env_label)_$(L - sys_block.length - 1).jld2")::Block{Nc}
            if engine <: GPUEngine
                env_trmat = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 1).jld2")::Vector{Matrix{Float64}})
            else
                env_trmat = load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 1).jld2")::Vector{Matrix{Float64}}
            end
        else
            env_block = block_table[env_label, L - sys_block.length - 1]
            if engine <: GPUEngine
                env_trmat = CuArray.(trmat_table[env_label, L - sys_block.length - 1])
            else
                env_trmat = trmat_table[env_label, L - sys_block.length - 1]
            end
        end

        env_tensor_dict = spin_operators!(tensor_table, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, fileio, scratch, dirid, block_table, trmat_table, on_the_fly, engine; lattice = lattice)
    else
        env_block = Block(L - sys_block.length - 1, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
        env_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
        env_trmat = Matrix{Float64}[]
    end

    env_block_enl = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)

    measurement = false
    Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0)

    for m in Iterators.flatten([m_sweep_list, Iterators.repeated(m_cooldown)])
        if rank == 0
            println("#")
            if measurement
                println("# Measurement step with (m, α) = ", m)
            else
                println("# Performing sweep with (m, α) = ", m)
            end
            println("#")
        end
        while true
            Ψ0_guess = eig_prediction(Ψ0, sys_label, sys_block_enl, env_block_enl, sys_trmat, env_trmat, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)

            if rank == 0
                if fileio
                    env_block = load_object("$scratch/temp$dirid/block_$(env_label)_$(L - sys_block.length - 2).jld2")::Block{Nc}
                    if engine <: GPUEngine
                        env_trmat = CuArray.(load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 2).jld2")::Vector{Matrix{Float64}})
                    else
                        env_trmat = load_object("$scratch/temp$dirid/trmat_$(env_label)_$(L - sys_block.length - 2).jld2")::Vector{Matrix{Float64}}
                    end
                else
                    env_block = block_table[env_label, L - sys_block.length - 2]
                    if engine <: GPUEngine
                        env_trmat = CuArray.(trmat_table[env_label, L - sys_block.length - 2])
                    else
                        env_trmat = trmat_table[env_label, L - sys_block.length - 2]
                    end
                end

                env_tensor_dict = spin_operators!(tensor_table, env_block, env_label, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, fileio, scratch, dirid, block_table, trmat_table, on_the_fly, engine; lattice = lattice)
            else
                env_block = Block(L - sys_block.length - 2, Tuple{Int, Int}[], SUNIrrep{Nc}[], Int[], Int[], Dict{Symbol, Vector{Matrix{Float64}}}())
                env_tensor_dict = Dict{Int, Matrix{Vector{Matrix{Float64}}}}()
                env_trmat = Matrix{Float64}[]
            end

            env_block_enl = enlarge_block(env_block, env_tensor_dict, Ly, widthmax, signfactor, comm, rank, Ncpu, tables, on_the_fly, engine; lattice = lattice)

            if env_block.length == 0
                Ψ0_guess = wavefunction_reverse(Ψ0_guess, sys_label, sys_block_enl, env_block_enl, widthmax, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine)

                sys_block, env_block = env_block, sys_block
                sys_tensor_dict, env_tensor_dict = env_tensor_dict, sys_tensor_dict
                sys_trmat, env_trmat = env_trmat, sys_trmat
                sys_block_enl, env_block_enl = env_block_enl, sys_block_enl
                sys_label, env_label = env_label, sys_label
            end

            if rank == 0
                println(graphic(sys_block, env_block; sys_label = sys_label))
            end

            cor = measurement && (sys_label == :r || sys_block.length == 0) ? correlation : :none
            sys_block, sys_tensor_dict, sys_block_enl, trerr, energy, Ψ0, sys_trmat, ee, es, Sj = dmrg_step!(SiSj, sys_label, sys_block, env_block, sys_tensor_dict, env_tensor_dict, sys_block_enl, env_block_enl, Ly, m..., widthmax, target, signfactor, comm, rank, Ncpu, tables, on_the_fly, γ_list, engine, Val(false); Ψ0_guess = Ψ0_guess, ES_max = ES_max, correlation = cor, margin = margin, lattice = lattice, Sj = Sj, alg = alg)

            if rank == 0
                if sys_label == :r && env_block_enl.length % Ly == 0
                    EE[env_block_enl.length ÷ Ly] = ee
                end
                println("E / N = ", energy / L)
                println("E     = ", energy)
                println("S_EE  = ", ee)
                if fileio
                    jldsave("$scratch/temp$dirid/block_$(sys_label)_$(sys_block.length).jld2"; env_block = sys_block)
                    jldsave("$scratch/temp$dirid/trmat_$(sys_label)_$(sys_block.length).jld2"; env_trmat = Array.(sys_trmat))
                else
                    block_table[sys_label, sys_block.length] = sys_block
                    trmat_table[sys_label, sys_block.length] = Array.(sys_trmat)
                end
            end

            if sys_label == :l && 2sys_block.length == L
                push!(m_list, m)
                push!(errors, trerr)
                push!(energies, energy)
                push!(EEs, ee)
                ES = es
                break
            end
        end

        if measurement
            break
        end

        if rank == 0
            if length(energies) > length(m_sweep_list) + 1
                measurement = abs((energies[end] - energies[end - 1]) / energies[end]) < tol_energy && abs((EEs[end] - EEs[end - 1]) / EEs[end]) < tol_EE
                MPI.bcast(measurement, 0, comm)
            end
        else
            if length(energies) > length(m_sweep_list) + 1
                measurement = MPI.bcast(nothing, 0, comm)::Bool
            end
        end
    end

    if rank == 0 && fileio
        rm("$scratch/temp$dirid"; recursive = true)
    end

    if engine <: GPUEngine
        magma_finalize()
    end

    MPI.Finalize()

    ESrtn = Dict{NTuple{Nc, Int}, Vector{Float64}}()
    for (key, value) in ES
        ESrtn[weight(key)] = value
    end

    if Nc == 2
        map!(x -> 0.5x, values(SiSj))
    end

    rank, DMRGOutput(m_list, errors, energies, EEs, EE, ESrtn, SiSj)
end