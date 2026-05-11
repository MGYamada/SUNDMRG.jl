"""
string = graphic(sys_block, env_block; sys_label = :l)
visualizes DMRG
"""
function graphic(sys_block, env_block; sys_label = :l)
    str = repeat("=", sys_block.length) * "**" * repeat("-", env_block.length)
    if sys_label == :r
        str = reverse(str)
    elseif sys_label != :l
        throw(ArgumentError("sys_label must be :l or :r"))
    end
    str
end

function _dmrg_step_blocks(sys_label, sys, env, sys_tensor_dict, env_tensor_dict, sys_enl, env_enl)
    _DMRGStepBlocks(sys_label, sys, env, sys_tensor_dict, env_tensor_dict, sys_enl, env_enl)
end

function _dmrg_step_settings(config::_FiniteRunConfig, runtime::_FiniteRuntime)
    (; Ly, widthmax, target, tables, alg, lattice) = config
    (; signfactor, on_the_fly, γ_list) = runtime
    _DMRGStepSettings(Ly, widthmax, target, signfactor, tables, on_the_fly, γ_list, alg, lattice)
end

function _dmrg_step_runtime(runtime::_FiniteRuntime)
    (; comm, rank, Ncpu, engine) = runtime
    _DMRGStepRuntime(comm, rank, Ncpu, engine)
end

function _dmrg_step_request(
    blocks,
    schedule::_DMRGSchedule,
    ::Val{env_calc};
    Ψ0_guess = nothing,
    ES_max = -Inf,
    correlation = Val(:none),
    margin = 0,
    Sj = Matrix{Vector{Matrix{Float64}}}(undef, 0, 0),
    noisy = true,
) where env_calc
    m, α = schedule
    options = _DMRGStepOptions(Ψ0_guess, ES_max, correlation, margin, Sj, noisy)
    _DMRGStepRequest(blocks, _DMRGStepSchedule(m, α), options, Val(env_calc))
end

function dmrg_step_result!(SiSj, request::_DMRGStepRequest, config::_FiniteRunConfig, runtime::_FiniteRuntime, ::Val{Nc}) where Nc
    dmrg_step!(SiSj, request, _dmrg_step_settings(config, runtime), _dmrg_step_runtime(runtime), Val(Nc))
end
