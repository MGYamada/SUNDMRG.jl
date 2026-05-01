abstract type AbstractInternalStorage end

struct MemoryInternalStorage{B, T, TT} <: AbstractInternalStorage
    block_table::B
    trmat_table::T
    tensor_table::TT
end

struct JLD2InternalStorage <: AbstractInternalStorage
    scratch::String
    dirid
end

function init_internal_storage(fileio, scratch, block_table, trmat_table, tensor_table, rank)
    if fileio
        dirid = rank == 0 ? lpad(rand(0 : 99999), 5, "0") : 0
        if rank == 0
            mkdir("$scratch/temp$dirid")
        end
        return JLD2InternalStorage(scratch, dirid)
    end
    return MemoryInternalStorage(block_table, trmat_table, tensor_table)
end

_block_filename(storage::JLD2InternalStorage, label, len) = "$(storage.scratch)/temp$(storage.dirid)/block_$(label)_$(len).jld2"
_trmat_filename(storage::JLD2InternalStorage, label, len) = "$(storage.scratch)/temp$(storage.dirid)/trmat_$(label)_$(len).jld2"
_tensor_filename(storage::JLD2InternalStorage, label, len, y) = "$(storage.scratch)/temp$(storage.dirid)/tensor_$(label)_$(len)_$(y).jld2"

function load_block(storage::MemoryInternalStorage, label, len)
    storage.block_table[label, len]
end

function save_block(storage::MemoryInternalStorage, label, len, block)
    storage.block_table[label, len] = block
end

function load_trmat(storage::MemoryInternalStorage, label, len)
    storage.trmat_table[label, len]
end

function save_trmat(storage::MemoryInternalStorage, label, len, trmat)
    storage.trmat_table[label, len] = trmat
end

function load_tensor(storage::MemoryInternalStorage, label, len, y)
    pop!(storage.tensor_table, (label, len, y), nothing)
end

function save_tensor(storage::MemoryInternalStorage, label, len, y, tensor)
    storage.tensor_table[label, len, y] = tensor
end

function load_block(storage::JLD2InternalStorage, label, len)
    load_object(_block_filename(storage, label, len))
end

function save_block(storage::JLD2InternalStorage, label, len, block)
    jldsave(_block_filename(storage, label, len); env_block = block)
end

function load_trmat(storage::JLD2InternalStorage, label, len)
    load_object(_trmat_filename(storage, label, len))::Vector{Matrix{Float64}}
end

function save_trmat(storage::JLD2InternalStorage, label, len, trmat)
    jldsave(_trmat_filename(storage, label, len); env_trmat = trmat)
end

function load_trmat(storage, label, len, engine)
    to_engine_array.(Ref(engine), load_trmat(storage, label, len))
end

function load_block_and_trmat(storage, label, len, engine, ::Val{Nc}) where Nc
    block = load_block(storage, label, len)::Block{Nc}
    trmat = load_trmat(storage, label, len, engine)
    return block, trmat
end

function save_block_and_trmat!(storage, label, block, trmat)
    save_block(storage, label, block.length, block)
    save_trmat(storage, label, block.length, Array.(trmat))
    return nothing
end

function load_tensor(storage::JLD2InternalStorage, label, len, y)
    filename = _tensor_filename(storage, label, len, y)
    if isfile(filename)
        tensor = load_object(filename)::Matrix{Vector{Matrix{Float64}}}
        rm(filename)
        tensor
    else
        nothing
    end
end

function save_tensor(storage::JLD2InternalStorage, label, len, y, tensor)
    jldsave(_tensor_filename(storage, label, len, y); env_tensor_dict = tensor)
end

function cleanup_storage!(storage::JLD2InternalStorage)
    rm("$(storage.scratch)/temp$(storage.dirid)"; recursive = true)
end

cleanup_storage!(::MemoryInternalStorage) = nothing
