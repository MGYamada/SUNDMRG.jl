module SUNDMRG

using MPI
using LinearAlgebra
using SUNRepresentations
using CUDA
using MAGMA

include("representation_theory.jl")
using .RepresentationTheory
using .RepresentationTheory: SparseVector2,
    sparsevec2,
    spzeros2,
    multiplicity,
    SYTdiagram,
    bf,
    subdiagram,
    P!,
    Papply!,
    Papply2!,
    representatives,
    antisymmetrize,
    SDC,
    _3ν,
    _6ν,
    _6νrev,
    _9ν,
    table_9ν

export DMRGOutput, run_DMRG, init_DMRG!, finalize_DMRG!, make_table, make_table3nu, make_table4, HeisenbergModel, SU, SquareLattice, HoneycombLattice, CPUEngine, GPUEngine, RepresentationTheory

include("types.jl")
include("lanczos.jl")
include("engine_utils.jl")
include("storage.jl")
include("block.jl")
include("measurement.jl")
include("step_types.jl")
include("step_workspace.jl")
include("step_density.jl")
include("step_lanczos.jl")
include("step_measurement.jl")
include("step.jl")
include("tools.jl")
include("runtime.jl")
include("finite_config.jl")
include("api.jl")
include("finite_support.jl")
include("finite_sweep.jl")
include("finite_phases.jl")
include("finite.jl")

end
