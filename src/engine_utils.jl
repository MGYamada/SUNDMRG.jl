"""
    zeros_like_engine(engine, T, dims...)

Allocate a zero-filled array with element type `T` and shape `dims`, matching the backend implied by `engine`.
"""
zeros_like_engine(::Type{<:CPUEngine}, T, dims...) = zeros(T, dims...)
zeros_like_engine(::Type{<:GPUEngine}, T, dims...) = CUDA.zeros(T, dims...)

"""
    to_engine_array(engine, x)

Convert `x` to an engine-native array representation.
"""
to_engine_array(::Type{<:CPUEngine}, x) = x
to_engine_array(::Type{<:GPUEngine}, x) = CuArray(x)

"""
    to_host_array(x)

Convert an engine array back to a host `Array` when needed.
"""
to_host_array(x) = x
to_host_array(x::CuArray) = Array(x)

"""
    engine_matrix_type(engine)

Return the concrete matrix type used by the selected engine for `Float64` data.
"""
engine_matrix_type(::Type{<:CPUEngine}) = Matrix{Float64}
engine_matrix_type(::Type{<:GPUEngine}) = CuMatrix{Float64}

empty_engine_matrix_vector(engine) = engine_matrix_type(engine)[]

function empty_engine_tensor_dict(engine)
    Dict{Int, Matrix{Vector{engine_matrix_type(engine)}}}()
end

function empty_engine_tensor_matrix(engine)
    Matrix{Vector{engine_matrix_type(engine)}}(undef, 0, 0)
end

function empty_engine_tensor_matrices(engine, n)
    [empty_engine_tensor_matrix(engine) for _ in 1 : n]
end

synchronize_engine(::Type{<:CPUEngine}) = nothing

function synchronize_engine(::Type{<:GPUEngine})
    CUDA.synchronize()
    return nothing
end

host_tensor_for_save(::Type{<:CPUEngine}, tensor, len) = deepcopy(tensor)
host_tensor_for_save(::Type{<:GPUEngine}, tensor, len) = [to_host_array.(tensor[ki, kj]) for ki in 1 : len, kj in 1 : len]

engine_tensor_from_host(::Type{<:CPUEngine}, tensor, len) = tensor
engine_tensor_from_host(engine::Type{<:GPUEngine}, tensor, len) = [to_engine_array.(Ref(engine), tensor[i, j]) for i in 1 : len, j in 1 : len]
