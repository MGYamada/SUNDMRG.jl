struct SuperBlock1CPU
    sys_H::Matrix{Float64}
    env_H::Matrix{Float64}
    sys_ind::Int
    env_ind::Int
end

struct SuperBlock1GPU
    sys_H::CuMatrix{Float64}
    env_H::CuMatrix{Float64}
    sys_ind::Int
    env_ind::Int
end

struct MiniBlock
    coeff::Float64
    om2::Int
    sys_τ::Int
    env_τ::Int
    sys_out::Int
    env_out::Int
end

struct SuperBlock2
    miniblock::Vector{MiniBlock}
    om1::Int
    sys_in::Int
    env_in::Int
    sys_in_size::Int
    env_in_size::Int
end

struct BlockEnlarging
    coeff::Float64
    τ1::Int
    τ2::Int
    range_i::UnitRange
    range_j::UnitRange
    i::Int
    j::Int
    ki::Int
    kj::Int
end

struct _StepSideContext
    len::Int
    ms
    betas
    dp
    conn::Int
    label::Symbol
    block
    block_enl
end

struct _StepLanczosContext
    target::Int
    comm
    rank::Int
    engine
    alg::Symbol
    superblock_H1
    bonds_hold
    x_conn::Int
    y_conn::Int
    sys_connS
    env_connS
    sys_ms
    env_ms
    sys_βs
    env_βs
    sys_dp
    env_dp
    sys_enlarge
    env_enlarge
    sys_tensor_dict_hold
    env_tensor_dict_hold
    superblock_H2
    OM
    sys_len::Int
    env_len::Int
    Ncpu::Int
end

struct _StepDensityContext
    comm
    rank::Int
    Ncpu::Int
    engine
    α::Float64
    m::Int
    ES_max::Float64
    noisy::Bool
end

struct _StepMeasurementContext
    SiSj
    Ly::Int
    x_conn::Int
    y_conn::Int
    sys_connS
    env_connS
    sys_len::Int
    env_len::Int
    sys_tensor_dict
    env_tensor_dict
    sys_tensor_dict_hold
    env_tensor_dict_hold
    sys_αs
    env_αs
    sys_βs
    env_βs
    sys_dp
    env_dp
    sys_ms
    env_ms
    sys_enlarge
    env_enlarge
    superblock_H2
    OM
    comm
    rank::Int
    Ncpu::Int
    engine
    correlation::Symbol
    margin::Int
    lattice::Symbol
end

struct _StepBlockContext
    Ly::Int
    widthmax::Int
    signfactor::Float64
    comm
    rank::Int
    Ncpu::Int
    tables
    on_the_fly::Bool
    engine
    lattice::Symbol
end

struct _StepCorrectionContext
    superblock_bonds
    sys_connS
    env_connS
    sys_enl
    env_enl
    x_conn::Int
    y_conn::Int
    sys_tensor_dict_hold
    env_tensor_dict_hold
    sys_tensor_dict
    env_tensor_dict
    sys_ms
    env_ms
    sys_dp
    env_dp
    sys_βs
    env_βs
    sys_enlarge
    env_enlarge
    engine
end

struct _StepWorkspace
    sys_αs
    env_αs
    sys_βs
    env_βs
    sys_ms
    env_ms
    sys_len::Int
    env_len::Int
    superblock_bonds
    bonds_hold
    x_conn::Int
    y_conn::Int
    sys_dp
    env_dp
    sys_enlarge
    env_enlarge
    OM
    superblock_H1
    superblock_H2
    sys_connS
    env_connS
    sys_tensor_dict_hold
    env_tensor_dict_hold
end
