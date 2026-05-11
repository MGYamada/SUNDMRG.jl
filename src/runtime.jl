struct _FiniteRuntime
    engine
    comm
    rank::Int
    Ncpu::Int
    on_the_fly
    mirror::Bool
    γ_type
    γ_list
    signfactor::Float64
end

_mode_value(x::Val{X}) where X = X
_mode_value(x) = x
_lattice_name(lattice) = _mode_value(lattice)
_correlation_name(correlation) = _mode_value(correlation)
_on_the_fly(on_the_fly) = _mode_value(on_the_fly)
_is_honeycomb_zc(lattice) = _lattice_name(lattice) == :honeycombZC
_is_square_lattice(lattice) = _lattice_name(lattice) == :square

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

isroot(rank::Integer) = rank == 0
isroot(runtime::_FiniteRuntime) = isroot(runtime.rank)

function root_println(rank::Integer, args...)
    if isroot(rank)
        println(args...)
    end
    return nothing
end

root_println(runtime::_FiniteRuntime, args...) = root_println(runtime.rank, args...)

function _init_runtime_and_engine(engine, lattice, Lx, Ly, Nc, rank, Ncpu)
    Nc >= 2 || throw(ArgumentError("Nc must be at least 2"))

    on_the_fly = Nc == 2
    mirror = lattice == :square || lattice == :honeycombZC

    iseven(Lx) || throw(ArgumentError("Lx must be even"))
    lattice ∈ (:square, :honeycombZC) || throw(ArgumentError("lattice must be :square or :honeycombZC"))
    if on_the_fly
        (Lx * Ly) % Nc == 0 || throw(ArgumentError("Lx * Ly must be divisible by Nc"))
    else
        if iseven(Nc)
            Ly % (Nc >> 1) == 0 || throw(ArgumentError("Ly must be divisible by Nc ÷ 2 for even Nc > 2"))
        else
            Ly % Nc == 0 || throw(ArgumentError("Ly must be divisible by Nc for odd Nc"))
        end
    end

    γ_type = typeof(trivialirrep(Val(Nc)))
    γ_list = γ_type[]
    for h in ((1 : Nc) .% Nc)
        push!(γ_list, SUNIrrep{Nc}(ntuple(i -> 0 + (i <= h), Val(Nc))))
    end

    _init_engine_runtime!(engine, rank, Ncpu)

    N = Lx * Ly
    signfactor = iseven(Nc) ? -1.0 : 1.0
    return Val(on_the_fly), mirror, γ_type, γ_list, N, signfactor
end

_init_engine_runtime!(::Type{<:CPUEngine}, rank, Ncpu) = nothing

function _init_engine_runtime!(::Type{<:GPUEngine}, rank, Ncpu)
    Ngpu = Int(length(devices()))
    Ncpu <= Ngpu || throw(ArgumentError("Ncpu must be less than or equal to the number of GPUs"))
    device!(rank)
    magma_init()
    return nothing
end

_finalize_engine_runtime!(::Type{<:CPUEngine}) = nothing

function _finalize_engine_runtime!(::Type{<:GPUEngine})
    magma_finalize()
    return nothing
end

function _finalize_runtime!(engine, ::Nothing, rank)
    _finalize_engine_runtime!(engine)
end

function _finalize_runtime!(engine, storage, rank)
    if rank == 0
        cleanup_storage!(storage)
    end

    _finalize_engine_runtime!(engine)
end
