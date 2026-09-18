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

"""
    init_DMRG!()

Initialize MPI for one or more DMRG runs.

Returns `true` when this call initialized MPI and `false` when MPI was already
initialized. Pair with [`finalize_DMRG!`](@ref) when calling [`run_DMRG`](@ref)
with `manage_mpi = false`.
"""
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

"""
    finalize_DMRG!()

Finalize MPI if it is currently initialized.

Returns `true` when MPI was finalized and `false` otherwise.
"""
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

# A guarded operation must contain no MPI communication. Every rank reaches
# this checkpoint before entering the next distributed phase. Keep local error
# identity/backtraces, and give peers the failing rank and its original message.
function _collective_local(f, comm, rank, Ncpu, phase)
    Ncpu == 1 && return f()
    result = try
        f()
    catch error
        _synchronize_local_failure(error, true, comm, rank, Ncpu, phase)
        rethrow()
    end
    _synchronize_local_failure(nothing, false, comm, rank, Ncpu, phase)
    return result
end

_collective_local(f, runtime::_FiniteRuntime, phase) =
    _collective_local(f, runtime.comm, runtime.rank, runtime.Ncpu, phase)

function _synchronize_local_failure(error, failed, comm, rank, Ncpu, phase)
    failed_rank = MPI.Allreduce(failed ? rank : Ncpu, MPI.MIN, comm)
    if failed_rank < Ncpu
        message = if rank == failed_rank
            try
                sprint(showerror, error)
            catch
                string(typeof(error))
            end
        else
            nothing
        end
        message = MPI.bcast(message, failed_rank, comm)::String
        if !failed
            throw(ErrorException("DMRG $phase failed on MPI rank $failed_rank: $message"))
        end
    end
    return nothing
end

function _collective_if_active(f, phase)
    if MPI.Initialized() && !MPI.Finalized()
        comm, rank, Ncpu = _comm_context()
        return _collective_local(f, comm, rank, Ncpu, phase)
    end
    return f()
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

function _runtime_parameters(lattice, Lx, Ly, Nc)
    Nc isa Integer && !(Nc isa Bool) || throw(ArgumentError("Nc must be an integer"))
    Nc >= 2 || throw(ArgumentError("Nc must be at least 2"))
    Lx = _positive_lattice_extent(Lx, "Lx")
    Ly = _positive_lattice_extent(Ly, "Ly")

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

    N = Lx * Ly
    signfactor = iseven(Nc) ? -1.0 : 1.0

    return Val(on_the_fly), mirror, γ_type, γ_list, N, signfactor
end

function _init_runtime_and_engine(engine, lattice, Lx, Ly, Nc, rank, Ncpu)
    comm = MPI.COMM_WORLD
    parameters = _collective_local(comm, rank, Ncpu, "runtime validation") do
        _runtime_parameters(lattice, Lx, Ly, Nc)
    end
    # GPU communicator setup is collective and must precede the local guard.
    context = _engine_runtime_context(engine, rank)
    acquired = false
    initialized = false
    return _with_cleanup(
        () -> begin
            _collective_local(comm, rank, Ncpu, "engine initialization") do
                _init_engine_runtime!(engine, rank, Ncpu, context)
                acquired = true
            end
            initialized = true
            return parameters
        end,
        () -> begin
            if !initialized
                _collective_local(comm, rank, Ncpu, "engine initialization rollback") do
                    acquired && _finalize_engine_runtime!(engine)
                end
            end
        end,
    )
end

_engine_runtime_context(::Type{<:CPUEngine}, rank) = nothing
_engine_runtime_context(::Type{<:GPUEngine}, rank) = _node_local_mpi_context(MPI.COMM_WORLD, rank)

_init_engine_runtime!(engine, rank, Ncpu, context) = _init_engine_runtime!(engine, rank, Ncpu)
_init_engine_runtime!(::Type{<:CPUEngine}, rank, Ncpu) = nothing

function _init_engine_runtime!(engine::Type{<:GPUEngine}, rank, Ncpu)
    return _init_engine_runtime!(engine, rank, Ncpu, _engine_runtime_context(engine, rank))
end

function _init_engine_runtime!(::Type{<:GPUEngine}, rank, Ncpu, context)
    local_rank, local_size = context
    Ngpu = Int(length(devices()))
    local_size <= Ngpu || throw(ArgumentError("the number of MPI processes on this node ($local_size) must not exceed the number of visible GPUs ($Ngpu)"))
    device!(local_rank)
    _init_magma_runtime!()
    return nothing
end

function _init_magma_runtime!(initialize = magma_init, finalize = magma_finalize)
    status = initialize()
    if status != MAGMA.MAGMA_SUCCESS
        # MAGMA increments its initialization count even when it returns an
        # error status. Roll back that reference before reporting the failure.
        # A call that throws before returning has not transferred ownership.
        return _with_cleanup(
            () -> throw(ErrorException("magma_init failed with status $status")),
            () -> _finalize_magma_runtime!(finalize),
        )
    end
    return nothing
end

function _finalize_magma_runtime!(finalize = magma_finalize)
    status = finalize()
    status == MAGMA.MAGMA_SUCCESS || throw(ErrorException("magma_finalize failed with status $status"))
    return nothing
end

function _node_local_mpi_context(comm, rank)
    local_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank)
    return _with_cleanup(
        () -> (MPI.Comm_rank(local_comm), MPI.Comm_size(local_comm)),
        () -> MPI.free(local_comm),
    )
end

_finalize_engine_runtime!(::Type{<:CPUEngine}) = nothing

function _finalize_engine_runtime!(::Type{<:GPUEngine})
    return _finalize_magma_runtime!()
end

function _finalize_runtime!(engine, ::Nothing, rank)
    _finalize_engine_runtime!(engine)
end

function _finalize_runtime!(engine, storage, rank)
    return _with_cleanup(
        () -> begin
            if rank == 0
                cleanup_storage!(storage)
            end
        end,
        () -> _finalize_engine_runtime!(engine),
    )
end
