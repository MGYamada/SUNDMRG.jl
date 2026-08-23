"""
    rank, dmrg = run_DMRG(model, lattice, m_warmup, m_sweep_list, m_cooldown, engine; kwargs...)

Run the finite-system DMRG algorithm.

`m_warmup`, `m_sweep_list`, and `m_cooldown` can be either integers, or
`(m, α)` tuples where `α` is the density-matrix mixing value.

Common keyword arguments:
- `target = 0`: target state (`0` for the ground state, `1` for the first excited state, etc.).
- `lanczos_maxiter = 100`: maximum Krylov basis size; must be at least `target + 1`.
- `widthmax = 0`, `tables = nothing`: representation table controls for SU(N > 2).
- `fileio = false`, `scratch = "."`: store intermediate blocks on disk.
- `correlation = :none`, `margin = 0`: optional correlation measurement.
- `tol_energy = 1e-5`, `tol_EE = 1e-3`, `max_cooldown_sweeps = 100`: convergence controls.
- `alg = :slow`: Lanczos/eigensolver mode.
- `verbose = true`: print progress information.
- `manage_mpi = true`: initialize/finalize MPI inside the call. Use `false` when MPI is managed externally.

Returns `(rank, dmrg)`. `dmrg` is a `DMRGOutput` on rank 0; nonzero MPI ranks
return `nothing` for the output.
"""
function run_DMRG end

function _dmrg_schedule(m::Integer)
    _dmrg_schedule(m, 0.0)
end

function _dmrg_schedule(m::Tuple{<:Integer, <:Real})
    _dmrg_schedule(m[1], m[2])
end

function _dmrg_schedule(m::Integer, α::Real)
    m > 0 || throw(ArgumentError("DMRG schedule bond dimension m must be positive"))
    α = Float64(α)
    isfinite(α) || throw(ArgumentError("DMRG schedule density-matrix mixing α must be finite"))
    α >= 0.0 || throw(ArgumentError("DMRG schedule density-matrix mixing α must be nonnegative"))
    return (Int(m), α)
end

_dmrg_schedule_list(ms::AbstractVector) = Tuple{Int, Float64}[_dmrg_schedule(m) for m in ms]

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::SquareLattice, m_warmup::Union{Integer, Tuple{<:Integer, <:Real}}, m_sweep_list::AbstractVector, m_cooldown::Union{Integer, Tuple{<:Integer, <:Real}}, engine::Type{<:Engine}; kwargs...) where Nc
    _run_DMRG(model, :square, lat.Lx, lat.Ly, _dmrg_schedule(m_warmup), _dmrg_schedule_list(m_sweep_list), _dmrg_schedule(m_cooldown), engine; kwargs...)
end

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::HoneycombLattice, m_warmup::Union{Integer, Tuple{<:Integer, <:Real}}, m_sweep_list::AbstractVector, m_cooldown::Union{Integer, Tuple{<:Integer, <:Real}}, engine::Type{<:Engine}; kwargs...) where Nc
    if lat.BC == :ZC
        return _run_DMRG(model, :honeycombZC, lat.Lx, lat.Ly, _dmrg_schedule(m_warmup), _dmrg_schedule_list(m_sweep_list), _dmrg_schedule(m_cooldown), engine; kwargs...)
    end
    throw(ArgumentError("HoneycombLattice with BC=$(lat.BC) is not supported (only :ZC)"))
end
