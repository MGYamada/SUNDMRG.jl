"""
    rank, dmrg = run_DMRG(model, lattice, m_warmup, m_sweep_list, m_cooldown, engine; kwargs...)

Run the finite-system DMRG algorithm.

`m_warmup`, `m_sweep_list`, and `m_cooldown` can be either integers, or
`(m, α)` tuples where `α` is the density-matrix mixing value.

Common keyword arguments:
- `target = 0`: target state (`0` for the ground state, `1` for the first excited state, etc.).
- `widthmax = 0`, `tables = nothing`: representation table controls for SU(N > 2).
- `fileio = false`, `scratch = "."`: store intermediate blocks on disk.
- `correlation = :none`, `margin = 0`: optional correlation measurement.
- `alg = :slow`: Lanczos/eigensolver mode.
- `verbose = true`: print progress information.
- `manage_mpi = true`: initialize/finalize MPI inside the call. Use `false` when MPI is managed externally.

Returns `(rank, dmrg)`. `dmrg` is a `DMRGOutput` on rank 0; nonzero MPI ranks
return `nothing` for the output.
"""
function run_DMRG end

_dmrg_schedule(m::Int) = (m, 0.0)
_dmrg_schedule(m::Tuple{Int, Float64}) = m
_dmrg_schedule_list(ms::AbstractVector) = Tuple{Int, Float64}[_dmrg_schedule(m) for m in ms]

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::SquareLattice, m_warmup::Union{Int, Tuple{Int, Float64}}, m_sweep_list::AbstractVector, m_cooldown::Union{Int, Tuple{Int, Float64}}, engine::Type{<:Engine}; kwargs...) where Nc
    _run_DMRG(model, :square, lat.Lx, lat.Ly, _dmrg_schedule(m_warmup), _dmrg_schedule_list(m_sweep_list), _dmrg_schedule(m_cooldown), engine; kwargs...)
end

function run_DMRG(model::HeisenbergModelSU{Nc}, lat::HoneycombLattice, m_warmup::Union{Int, Tuple{Int, Float64}}, m_sweep_list::AbstractVector, m_cooldown::Union{Int, Tuple{Int, Float64}}, engine::Type{<:Engine}; kwargs...) where Nc
    if lat.BC == :ZC
        return _run_DMRG(model, :honeycombZC, lat.Lx, lat.Ly, _dmrg_schedule(m_warmup), _dmrg_schedule_list(m_sweep_list), _dmrg_schedule(m_cooldown), engine; kwargs...)
    end
    throw(ArgumentError("HoneycombLattice with BC=$(lat.BC) is not supported (only :ZC)"))
end
