"""
    zeros_like_engine(engine, T, dims...)

Allocate a zero-filled array with element type `T` and shape `dims`, matching the backend implied by `engine`.
"""
function zeros_like_engine(engine, T, dims...)
    if engine <: GPUEngine
        CUDA.zeros(T, dims...)
    else
        zeros(T, dims...)
    end
end

"""
    to_engine_array(engine, x)

Convert `x` to an engine-native array representation.
"""
function to_engine_array(engine, x)
    if engine <: GPUEngine
        CuArray(x)
    else
        x
    end
end

"""
    to_host_array(x)

Convert an engine array back to a host `Array` when needed.
"""
function to_host_array(x)
    if x isa CuArray
        Array(x)
    else
        x
    end
end

"""
    engine_matrix_type(engine)

Return the concrete matrix type used by the selected engine for `Float64` data.
"""
function engine_matrix_type(engine)
    if engine <: GPUEngine
        CuMatrix{Float64}
    else
        Matrix{Float64}
    end
end

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

function synchronize_engine(engine)
    if engine <: GPUEngine
        CUDA.synchronize()
    end
    return nothing
end
