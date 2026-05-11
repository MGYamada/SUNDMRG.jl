const _DMRGSchedule = Tuple{Int, Float64}

struct _FiniteRunConfig{L,T,S,C,A}
    lattice::L
    Lx::Int
    Ly::Int
    N::Int
    Nc::Int
    m_warmup::_DMRGSchedule
    m_sweep_list::Vector{_DMRGSchedule}
    m_cooldown::_DMRGSchedule
    target::Int
    widthmax::Int
    tables::T
    fileio::Bool
    scratch::S
    ES_max::Float64
    tol_energy::Float64
    tol_EE::Float64
    correlation::C
    margin::Int
    alg::A
    verbose::Bool
end

mutable struct _FiniteState{Nc,StorageT,BLT,TLT,BRT,TRT,BELT,BERT,TrLT,TrRT,ΨT}
    m_list::Vector{_DMRGSchedule}
    errors::Vector{Float64}
    energies::Vector{Float64}
    EEs::Vector{Float64}
    EE::Vector{Float64}
    ES::Dict{SUNIrrep{Nc}, Vector{Float64}}
    SiSj::Dict{Tuple{Int, Int}, Float64}
    storage::StorageT
    blockL::BLT
    blockL_tensor_dict::TLT
    blockR::BRT
    blockR_tensor_dict::TRT
    blockL_enl::BELT
    blockR_enl::BERT
    trmatL::TrLT
    trmatR::TrRT
    Ψ::ΨT
end

mutable struct _GrowthState{BlocksT,TensorsT,TrmatsT,EnlsT}
    L::Int
    sys_blocks::BlocksT
    sys_tensor_dicts::TensorsT
    sys_trmats::TrmatsT
    sys_block_enls::EnlsT
end
