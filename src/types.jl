"""
    DMRGOutput{Nc}

Finite-system DMRG results returned on rank 0.

Fields:
- `ms`: kept SU(N) multiplets and density-matrix mixing values used at each recorded step.
- `errors`: truncation error for each recorded step.
- `energies`: total energy for each recorded step.
- `EEs`: entanglement entropy at the active cut for each recorded step.
- `EE`: entanglement entropy profile measured during the final converged sweep.
- `ES`: entanglement spectrum by SU(N) irrep, keyed by irrep weight.
- `SiSj`: measured two-site correlation values keyed by site pair.
"""
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

"""
    HeisenbergModel()

Nearest-neighbor Heisenberg model marker used together with an SU symmetry,
for example `SU(2)HeisenbergModel()`.
"""
struct HeisenbergModel <: Model end

abstract type Symmetry end

"""
    SU(Nc)

Construct an SU(Nc) symmetry marker.

`Nc` must be an integer greater than or equal to 2.

Use it by multiplying with a model marker, either explicitly as
`SU(3) * HeisenbergModel()` or with Julia's juxtaposition syntax:
`SU(3)HeisenbergModel()`.
"""
struct SU{Nc} <: Symmetry end

function SU(Nc)
    Nc isa Integer && !(Nc isa Bool) || throw(ArgumentError("Nc must be an integer"))
    Nc >= 2 || throw(ArgumentError("Nc must be at least 2"))
    return SU{_checked_int(Nc, "Nc")}()
end

abstract type SymmetricModel{S, M} <: Model where {S <: Symmetry, M <: Model} end

"""
    SU(Nc)HeisenbergModel()

SU(Nc)-symmetric Heisenberg model marker accepted by [`run_DMRG`](@ref).
"""
struct HeisenbergModelSU{Nc} <: SymmetricModel{SU{Nc}, HeisenbergModel} end

Base.:*(::SU{Nc}, ::HeisenbergModel) where Nc = HeisenbergModelSU{Nc}()

abstract type Lattice{D} end

"""
    SquareLattice(Lx, Ly)

Two-dimensional square lattice with `Lx` columns and `Ly` legs.

Both extents must be positive integers.
"""
struct SquareLattice <: Lattice{2}
    Lx::Int
    Ly::Int
    function SquareLattice(Lx, Ly)
        new(_positive_lattice_extent(Lx, "Lx"), _positive_lattice_extent(Ly, "Ly"))
    end
end

"""
    HoneycombLattice(Lx, Ly, :ZC)

Two-dimensional honeycomb cylinder with zigzag-cylinder boundary condition.

Both extents must be positive integers, and `Ly` must be even. `:ZC` is
currently the only supported boundary condition.
"""
struct HoneycombLattice <: Lattice{2}
    Lx::Int
    Ly::Int
    BC::Symbol
    function HoneycombLattice(Lx, Ly, BC)
        Lx = _positive_lattice_extent(Lx, "Lx")
        Ly = _positive_lattice_extent(Ly, "Ly")
        iseven(Ly) || throw(ArgumentError("Ly must be even for HoneycombLattice"))
        BC == :ZC || throw(ArgumentError("HoneycombLattice only supports BC = :ZC"))
        new(Lx, Ly, BC)
    end
end

function _checked_int(value::Integer, name)
    try
        return Int(value)
    catch err
        err isa InexactError || rethrow()
        throw(ArgumentError("$name is outside the supported Int range"))
    end
end

function _positive_lattice_extent(value, name)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$name must be an integer"))
    value > 0 || throw(ArgumentError("$name must be positive"))
    return _checked_int(value, name)
end

abstract type Engine end

"""
    CPUEngine

Execution backend that stores dense arrays on the CPU.
"""
abstract type CPUEngine <: Engine end

"""
    GPUEngine

Execution backend that stores dense arrays on CUDA devices.
"""
abstract type GPUEngine <: Engine end
