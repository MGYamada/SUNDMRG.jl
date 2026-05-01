struct _FiniteRuntime
    engine
    comm
    rank::Int
    Ncpu::Int
    on_the_fly::Bool
    mirror::Bool
    γ_type
    γ_list
    signfactor::Float64
end

function init_DMRG!()
    if MPI.Finalized()
        throw(ArgumentError("MPI has already been finalized and cannot be initialized again in this process"))
    end
    if !MPI.Initialized()
        MPI.Init(; threadlevel = MPI.THREAD_FUNNELED)
        return true
    end
    return false
end

function finalize_DMRG!()
    if MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
        return true
    end
    return false
end

function _comm_context()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Ncpu = MPI.Comm_size(comm)
    return comm, rank, Ncpu
end

function _init_runtime_and_engine(engine, lattice, Lx, Ly, Nc, rank, Ncpu)
    @assert Nc >= 2

    on_the_fly = Nc == 2
    mirror = lattice == :square || lattice == :honeycombZC

    @assert iseven(Lx)
    @assert lattice == :square || lattice == :honeycombZC
    if on_the_fly
        @assert (Lx * Ly) % Nc == 0
    else
        if iseven(Nc)
            @assert Ly % (Nc >> 1) == 0
        else
            @assert Ly % Nc == 0
        end
    end

    γ_type = typeof(trivialirrep(Val(Nc)))
    γ_list = γ_type[]
    for h in ((1 : Nc) .% Nc)
        push!(γ_list, SUNIrrep{Nc}(ntuple(i -> 0 + (i <= h), Val(Nc))))
    end

    if engine <: GPUEngine
        Ngpu = Int(length(devices()))
        @assert Ncpu <= Ngpu
        device!(rank)
        magma_init()
    end

    N = Lx * Ly
    signfactor = iseven(Nc) ? -1.0 : 1.0
    return on_the_fly, mirror, γ_type, γ_list, N, signfactor
end

function _finalize_runtime!(engine, storage, rank)
    if rank == 0
        cleanup_storage!(storage)
    end

    if engine <: GPUEngine
        magma_finalize()
    end
end
