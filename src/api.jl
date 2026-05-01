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
