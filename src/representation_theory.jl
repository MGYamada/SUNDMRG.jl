module RepresentationTheory

using LinearAlgebra
using MPI
using SUNRepresentations
using RationalRoots
using WignerSymbols
using JLD2
using ThreadsX
using Permutations
using OMEinsum

export RepresentationTable,
    irrep,
    irreplist,
    outer_multiplicity,
    OM_matrix,
    trivialirrep,
    adjointirrep,
    fundamentalirrep,
    wigner3ν,
    wigner6ν,
    racahU,
    wigner9ν,
    wigner9j,
    wigner6νrev,
    make_table,
    make_table3nu,
    make_table4

"""
    RepresentationTheory

Submodule containing SU(N) representation helpers and coefficient-table
generation routines used by SUNDMRG.
"""
RepresentationTheory

"""
    RepresentationTable

Tuple type used for the combined SU(N) coefficient tables consumed by DMRG
runs.
"""
const RepresentationTable = NTuple{6, Any}

function _validate_table_parameters(Nc, widthmax)
    Nc isa Integer && !(Nc isa Bool) || throw(ArgumentError("Nc must be an integer"))
    Nc >= 2 || throw(ArgumentError("Nc must be at least 2"))
    widthmax isa Integer && !(widthmax isa Bool) || throw(ArgumentError("widthmax must be an integer"))
    widthmax >= 0 || throw(ArgumentError("widthmax must be nonnegative"))

    try
        return Int(Nc), Int(widthmax)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("Nc and widthmax must fit in Int"))
    end
end

function _init_table_mpi!(manage_mpi::Bool)
    if !manage_mpi
        (MPI.Initialized() && !MPI.Finalized()) || throw(ArgumentError("MPI must be initialized when manage_mpi = false"))
        return false
    end
    if MPI.Finalized()
        throw(ArgumentError("MPI has already been finalized and cannot be initialized again in this process"))
    end
    if !MPI.Initialized()
        MPI.Init(; threadlevel = MPI.THREAD_FUNNELED)
        return true
    end
    return false
end

function _finalize_table_mpi!(did_initialize::Bool)
    if did_initialize && MPI.Initialized() && !MPI.Finalized()
        MPI.Finalize()
        return true
    end
    return false
end

function _with_table_mpi(f::Function, manage_mpi::Bool)
    did_initialize_mpi = _init_table_mpi!(manage_mpi)
    try
        return f()
    finally
        _finalize_table_mpi!(did_initialize_mpi)
    end
end

include("sparsevec2.jl")
include("suncalc.jl")
include("sytx.jl")
include("subduction.jl")
include("tablecalc.jl")
include("table4.jl")
include("table3nu.jl")
include("table.jl")

end
