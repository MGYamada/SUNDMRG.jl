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
struct HeisenbergModel <: Model end

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
        iseven(Ly) || throw(ArgumentError("Ly must be even for HoneycombLattice"))
        BC == :ZC || throw(ArgumentError("HoneycombLattice only supports BC = :ZC"))
        new(Lx, Ly, BC)
    end
end

abstract type Engine end
abstract type CPUEngine <: Engine end
abstract type GPUEngine <: Engine end
