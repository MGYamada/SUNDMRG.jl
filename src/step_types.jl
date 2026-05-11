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
    range_i::UnitRange{Int}
    range_j::UnitRange{Int}
    i::Int
    j::Int
    ki::Int
    kj::Int
end

struct _StepSideContext{MS,B,D,BT,BET}
    len::Int
    ms::MS
    betas::B
    dp::D
    conn::Int
    label::Symbol
    block::BT
    block_enl::BET
end

struct _StepLanczosContext{CommT,E,A,H1,BondsT,SysConnT,EnvConnT,MST,BetaT,DPT,EnlargeT,SysTensorT,EnvTensorT,H2T,OMT}
    target::Int
    comm::CommT
    rank::Int
    engine::Type{E}
    alg::A
    superblock_H1::H1
    bonds_hold::BondsT
    x_conn::Int
    y_conn::Int
    sys_connS::SysConnT
    env_connS::EnvConnT
    sys_ms::MST
    env_ms::MST
    sys_βs::BetaT
    env_βs::BetaT
    sys_dp::DPT
    env_dp::DPT
    sys_enlarge::EnlargeT
    env_enlarge::EnlargeT
    sys_tensor_dict_hold::SysTensorT
    env_tensor_dict_hold::EnvTensorT
    superblock_H2::H2T
    OM::OMT
    sys_len::Int
    env_len::Int
    Ncpu::Int
end

struct _StepDensityContext{CommT,E}
    comm::CommT
    rank::Int
    Ncpu::Int
    engine::Type{E}
    α::Float64
    m::Int
    ES_max::Float64
    noisy::Bool
end

struct _StepMeasurementContext{SiSjT,SysConnT,EnvConnT,SysTensorT,EnvTensorT,SysTensorHoldT,EnvTensorHoldT,BetaT,DPT,MST,EnlargeT,H2T,OMT,CommT,E,C,L}
    SiSj::SiSjT
    Ly::Int
    x_conn::Int
    y_conn::Int
    sys_connS::SysConnT
    env_connS::EnvConnT
    sys_len::Int
    env_len::Int
    sys_tensor_dict::SysTensorT
    env_tensor_dict::EnvTensorT
    sys_tensor_dict_hold::SysTensorHoldT
    env_tensor_dict_hold::EnvTensorHoldT
    sys_αs::BetaT
    env_αs::BetaT
    sys_βs::BetaT
    env_βs::BetaT
    sys_dp::DPT
    env_dp::DPT
    sys_ms::MST
    env_ms::MST
    sys_enlarge::EnlargeT
    env_enlarge::EnlargeT
    superblock_H2::H2T
    OM::OMT
    comm::CommT
    rank::Int
    Ncpu::Int
    engine::Type{E}
    correlation::C
    margin::Int
    lattice::L
end

struct _StepBlockContext{CommT,TablesT,O,E,L}
    Ly::Int
    widthmax::Int
    signfactor::Float64
    comm::CommT
    rank::Int
    Ncpu::Int
    tables::TablesT
    on_the_fly::O
    engine::Type{E}
    lattice::L
end

struct _StepCorrectionContext{BondsT,SysConnT,EnvConnT,SysEnlT,EnvEnlT,SysTensorHoldT,EnvTensorHoldT,SysTensorT,EnvTensorT,MST,DPT,BetaT,EnlargeT,E}
    superblock_bonds::BondsT
    sys_connS::SysConnT
    env_connS::EnvConnT
    sys_enl::SysEnlT
    env_enl::EnvEnlT
    x_conn::Int
    y_conn::Int
    sys_tensor_dict_hold::SysTensorHoldT
    env_tensor_dict_hold::EnvTensorHoldT
    sys_tensor_dict::SysTensorT
    env_tensor_dict::EnvTensorT
    sys_ms::MST
    env_ms::MST
    sys_dp::DPT
    env_dp::DPT
    sys_βs::BetaT
    env_βs::BetaT
    sys_enlarge::EnlargeT
    env_enlarge::EnlargeT
    engine::Type{E}
end

struct _StepWorkspace{SysAlphaT,EnvAlphaT,SysBetaT,EnvBetaT,SysMST,EnvMST,BondsT,HoldBondsT,SysDPT,EnvDPT,SysEnlargeT,EnvEnlargeT,OMT,H1T,H2T,SysConnT,EnvConnT,SysTensorT,EnvTensorT}
    sys_αs::SysAlphaT
    env_αs::EnvAlphaT
    sys_βs::SysBetaT
    env_βs::EnvBetaT
    sys_ms::SysMST
    env_ms::EnvMST
    sys_len::Int
    env_len::Int
    superblock_bonds::BondsT
    bonds_hold::HoldBondsT
    x_conn::Int
    y_conn::Int
    sys_dp::SysDPT
    env_dp::EnvDPT
    sys_enlarge::SysEnlargeT
    env_enlarge::EnvEnlargeT
    OM::OMT
    superblock_H1::H1T
    superblock_H2::H2T
    sys_connS::SysConnT
    env_connS::EnvConnT
    sys_tensor_dict_hold::SysTensorT
    env_tensor_dict_hold::EnvTensorT
end
