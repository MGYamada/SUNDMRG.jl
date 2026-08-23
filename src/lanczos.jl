const tol_wavefunction = 1e-13
const tol_Lanczos = 1e-13

MyMatrix = Matrix{<:Vector{<:AbstractMatrix{Float64}}}

function mydot(x::MyMatrix, y::MyMatrix)
    s = 0.0
    for (IX, IY) in zip(eachindex(x), eachindex(y))
        for (JX, JY) in zip(eachindex(x[IX]), eachindex(y[IY]))
            s += LinearAlgebra.dot(x[IX][JX], y[IY][JY])
        end
    end
    s
end

function myaxpy!(α, x::MyMatrix, y::MyMatrix)
    for (IY, IX) in zip(eachindex(y), eachindex(x))
        for (JY, JX) in zip(eachindex(y[IY]), eachindex(x[IX]))
            @. y[IY][JY] += α * x[IX][JX]
        end
    end
    y
end

function myaxpby!(α, x::MyMatrix, β, y::MyMatrix)
    for (IX, IY) in zip(eachindex(x), eachindex(y))
        for (JX, JY) in zip(eachindex(x[IX]), eachindex(y[IY]))
            @. y[IY][JY] = α * x[IX][JX] + β * y[IY][JY]
        end
    end
    y
end

function myrmul!(A::MyMatrix, b::Number)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] .*= b
        end
    end
    A
end

function myrdiv!(A::MyMatrix, b::Number)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] ./= b
        end
    end
    A
end

function mycopyto!(dest::MyMatrix, src::MyMatrix)
    for (IX, IY) in zip(eachindex(dest), eachindex(src))
        for (JX, JY) in zip(eachindex(dest[IX]), eachindex(src[IY]))
            dest[IX][JX] .= src[IY][JY]
        end
    end
    dest
end

function myzero!(A::MyMatrix)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] .= 0.0
        end
    end
    A
end

"""
CG!(A!, val, x, Ax, buffer1, buffer2, comm, rank)
CG routine for the Lanczos method
"""
function CG!(A!::Function, val, x, Ax, buffer1, buffer2, comm, rank)
    valnew = 0.0
    r = Ax
    p = buffer1
    Ap = buffer2
    for i in 1 : 100
        valshift = val - 1e-8
        myaxpby!(1.0 + valshift, x, -1.0, r)
        mycopyto!(p, r)
        normold = MPI.Allreduce(mydot(r, r), MPI.SUM, comm)
        j = 1
        while true
            myzero!(Ap)
            A!(Ap, p)
            myaxpy!(-valshift, p, Ap)
            α = normold / MPI.Allreduce(mydot(p, Ap), MPI.SUM, comm)
            myaxpy!(α, p, x)
            myaxpy!(-α, Ap, r)
            normnew = MPI.Allreduce(mydot(r, r), MPI.SUM, comm)
            if j == 10 || normnew < 1e-8
                break
            end
            β = normnew / normold
            myaxpby!(1.0, r, β, p)
            normold = normnew
            j += 1
        end
        myrdiv!(x, sqrt(MPI.Allreduce(mydot(x, x), MPI.SUM, comm)))
        myzero!(r)
        A!(r, x)
        valnew = MPI.Allreduce(mydot(x, r), MPI.SUM, comm)
        if _lanczos_value_converged(valnew, val)
            break
        end
        val = valnew
    end
    valnew
end

"""
Lanczos!(A!, initial, position, comm, rank, engine; maxiter = 100, alg = :slow, allow_fewer = false)
returns eigenpairs of a linear map A!
"""
function Lanczos!(A!::Function, initial, position, comm, rank, engine; maxiter = 100, alg = :slow, allow_fewer = false)
    Lanczos!(A!, initial, position, comm, rank, engine, _alg_value(alg); maxiter = maxiter, allow_fewer = allow_fewer)
end

_alg_value(alg::Symbol) = Val(alg)
_alg_value(alg::Val) = alg

function Lanczos!(A!::Function, initial, position, comm, rank, engine, mode::Val; maxiter = 100, allow_fewer = false)
    _validate_lanczos_request(position, maxiter, allow_fewer)
    ketkm1 = _zero_lanczos_vector(initial)
    initial_norm = sqrt(MPI.Allreduce(mydot(initial, initial), MPI.SUM, comm))
    isfinite(initial_norm) && initial_norm > 0.0 || throw(ArgumentError("Lanczos initial vector must have a positive finite norm"))
    myrdiv!(initial, initial_norm)
    ketk = deepcopy(initial)
    ketk1 = deepcopy(ketk)
    β = 0.0
    αlist = Float64[]
    βlist = Float64[]
    vals = Float64[]
    vecs = zeros(0, 0)
    basis = position > 1 ? [deepcopy(ketk)] : nothing
    cache = basis === nothing ? _init_lanczos_cache(mode, ketk) : nothing
    k = 1
    while true
        myzero!(ketk1)
        A!(ketk1, ketk)
        α = MPI.Allreduce(mydot(ketk, ketk1), MPI.SUM, comm)
        push!(αlist, α)
        myaxpy!(-β, ketkm1, ketk1)
        myaxpy!(-α, ketk, ketk1)

        if basis !== nothing
            _reorthogonalize_lanczos!(ketk1, basis, comm)
        end

        βnew = sqrt(MPI.Allreduce(mydot(ketk1, ketk1), MPI.SUM, comm))
        isfinite(βnew) || throw(ErrorException("Lanczos residual norm is not finite"))
        breakdown = _lanczos_breakdown(βnew, α, β)

        if breakdown && k < position && _restart_lanczos_chain!(ketk1, basis, comm, engine)
            myzero!(ketkm1)
            mycopyto!(ketk, ketk1)
            β = 0.0
            push!(βlist, β)
            push!(basis, deepcopy(ketk))
            k += 1
            continue
        end

        if k >= position || breakdown || k == maxiter
            vals, vecs = _lanczos_ritz_vectors(αlist, βlist, comm, rank)
        end

        converged = k >= position && _lanczos_ritz_converged(vals, vecs, position, βnew, breakdown, comm, rank)
        if breakdown || converged || k == maxiter
            break
        end

        myrdiv!(ketk1, βnew)
        mycopyto!(ketkm1, ketk)
        mycopyto!(ketk, ketk1)
        β = βnew
        push!(βlist, β)
        if basis === nothing
            _cache_lanczos_vector!(mode, cache, ketk)
        else
            push!(basis, deepcopy(ketk))
        end
        k += 1
    end
    vecs = MPI.bcast(vecs, 0, comm)::Matrix{Float64}
    if size(vecs, 2) == 0
        throw(ErrorException("Lanczos did not produce any Ritz pairs"))
    end

    requested_position = position
    available_positions = size(vecs, 2)
    if requested_position > available_positions
        allow_fewer || throw(ArgumentError("Lanczos requested eigenposition $requested_position, but only $available_positions eigenpairs are available"))
        position = available_positions
    end

    if basis === nothing
        _reconstruct_lanczos_vector!(mode, A!, engine, initial, ketk, ketk1, ketkm1, vecs, αlist, βlist, position, cache)
    else
        _reconstruct_lanczos_vector_from_basis!(initial, basis, vecs, position)
    end
    _refine_lanczos_vector!(A!, initial, ketk, ketkm1, ketk1, vals[position], position, maxiter, comm, rank)
end

function _validate_lanczos_request(position, maxiter, allow_fewer = false)
    position isa Integer || throw(ArgumentError("Lanczos position must be an integer"))
    position > 0 || throw(ArgumentError("Lanczos position must be positive"))
    maxiter isa Integer || throw(ArgumentError("Lanczos maxiter must be an integer"))
    maxiter > 0 || throw(ArgumentError("Lanczos maxiter must be positive"))
    position <= maxiter || throw(ArgumentError("Lanczos position must not exceed maxiter"))
    allow_fewer isa Bool || throw(ArgumentError("Lanczos allow_fewer must be true or false"))
    return nothing
end

function _zero_lanczos_vector(initial)
    ket = deepcopy(initial)
    return myzero!(ket)
end

function _lanczos_ritz_vectors(αlist, βlist, comm, rank)
    vals = Float64[]
    vecs = zeros(0, 0)
    if rank == 0
        vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
    end
    vals = MPI.bcast(vals, 0, comm)::Vector{Float64}
    return vals, vecs
end

_lanczos_breakdown(βnew, α, βold) = βnew <= tol_Lanczos * max(abs(α), abs(βold), 1.0)
_lanczos_ritz_residual_converged(val, last_component, βnew) = abs(βnew * last_component) <= tol_Lanczos * max(abs(val), 1.0)
_lanczos_value_converged(current, previous) = isfinite(current) && isfinite(previous) && abs(current - previous) <= tol_wavefunction * max(abs(current), abs(previous), 1.0)
_lanczos_residual_converged(val, var) = isfinite(val) && isfinite(var) && abs(var - val ^ 2) <= tol_wavefunction * max(abs(val ^ 2), 1.0)

function _lanczos_refinement_converged(val, target_val, var, position)
    _lanczos_residual_converged(val, var) || return false
    position == 1 && return true
    return _lanczos_value_converged(val, target_val)
end

function _lanczos_ritz_converged(vals, vecs, position, βnew, breakdown, comm, rank)
    converged = rank == 0 ? breakdown || _lanczos_ritz_residual_converged(vals[position], vecs[end, position], βnew) : nothing
    return MPI.bcast(converged, 0, comm)::Bool
end

function _reorthogonalize_lanczos!(vector, basis, comm)
    for _ in 1 : 2
        for basis_vector in basis
            coefficient = MPI.Allreduce(mydot(basis_vector, vector), MPI.SUM, comm)
            myaxpy!(-coefficient, basis_vector, vector)
        end
    end
    return vector
end

function _restart_lanczos_chain!(vector, basis, comm, engine)
    basis === nothing && return false
    for _ in 1 : 8
        _randomize_lanczos_vector!(vector, engine)
        raw_norm = sqrt(MPI.Allreduce(mydot(vector, vector), MPI.SUM, comm))
        _reorthogonalize_lanczos!(vector, basis, comm)
        restart_norm = sqrt(MPI.Allreduce(mydot(vector, vector), MPI.SUM, comm))
        if isfinite(restart_norm) && restart_norm > sqrt(eps(Float64)) * max(raw_norm, 1.0)
            myrdiv!(vector, restart_norm)
            return true
        end
    end
    return false
end

function _randomize_lanczos_vector!(vector, engine)
    for I in eachindex(vector), J in eachindex(vector[I])
        vector[I][J] .= _random_lanczos_matrix(engine, size(vector[I][J])...)
    end
    return vector
end

_random_lanczos_matrix(::Type{<:CPUEngine}, dimensions...) = rand(Float64, dimensions...)
_random_lanczos_matrix(::Type{<:GPUEngine}, dimensions...) = CUDA.rand(Float64, dimensions...)

function _reconstruct_lanczos_vector_from_basis!(initial, basis, vecs, position)
    myzero!(initial)
    for k in axes(vecs, 1)
        myaxpy!(vecs[k, position], basis[k], initial)
    end
    return position
end

function _reconstruct_lanczos_vector!(mode::Val, A!, engine, initial, ketk, ketk1, ketkm1, vecs, αlist, βlist, position, cache)
    myzero!(ketkm1)
    mycopyto!(ketk, initial)
    β = 0.0
    myrmul!(initial, vecs[1, position])
    for k in 1 : size(vecs, 1) - 1
        β = _lanczos_reconstruct_step!(mode, A!, engine, vecs[k + 1, position], cache, initial, ketk, ketk1, ketkm1, αlist, βlist, k, β)
    end
    return position
end

function _refine_lanczos_vector!(A!, initial, ketk, ketkm1, ketk1, target_val, position, maxiter, comm, rank)
    norm_initial = sqrt(MPI.Allreduce(mydot(initial, initial), MPI.SUM, comm))
    isfinite(norm_initial) && norm_initial > 0.0 || throw(ErrorException("Lanczos reconstructed vector has a non-positive or non-finite norm"))
    myrdiv!(initial, norm_initial)
    val, var = _lanczos_wavefunction_metrics!(A!, initial, ketk, comm)
    if _lanczos_refinement_converged(val, target_val, var, position)
        return val
    end

    CG!(A!, val, initial, ketk, ketkm1, ketk1, comm, rank)
    val, var = _lanczos_wavefunction_metrics!(A!, initial, ketk, comm)
    if !_lanczos_refinement_converged(val, target_val, var, position)
        residual = sqrt(abs(var - val ^ 2))
        throw(ErrorException("Lanczos refinement failed to converge eigenposition $position after at most $maxiter Krylov iterations (residual norm = $residual)"))
    end
    return val
end

function _lanczos_wavefunction_metrics!(A!, vector, image, comm)
    myzero!(image)
    A!(image, vector)
    val = MPI.Allreduce(mydot(vector, image), MPI.SUM, comm)
    var = MPI.Allreduce(mydot(image, image), MPI.SUM, comm)
    return val, var
end

_init_lanczos_cache(::Val{:slow}, ketk) = nothing

function _init_lanczos_cache(::Val{:fast}, ketk)
    m, n = size(ketk)
    return (m = m, n = n, vectors = Matrix{Vector{Matrix{Float64}}}[])
end

_cache_lanczos_vector!(::Val{:slow}, cache, ketk) = nothing

function _cache_lanczos_vector!(::Val{:fast}, cache, ketk)
    push!(cache.vectors, map(x -> Array.(x), ketk))
    return nothing
end

function _lanczos_reconstruct_step!(::Val{:slow}, A!, engine, coeff, cache, initial, ketk, ketk1, ketkm1, αlist, βlist, k, β)
    myzero!(ketk1)
    A!(ketk1, ketk)
    α = αlist[k]
    myaxpy!(-β, ketkm1, ketk1)
    myaxpy!(-α, ketk, ketk1)
    β = βlist[k]
    myrdiv!(ketk1, β)
    mycopyto!(ketkm1, ketk)
    mycopyto!(ketk, ketk1)
    myaxpy!(coeff, ketk, initial)
    return β
end

function _lanczos_reconstruct_step!(::Val{:fast}, A!, engine, coeff, cache, initial, ketk, ketk1, ketkm1, αlist, βlist, k, β)
    _fast_lanczos_axpy!(engine, coeff, cache.vectors[k], initial, cache.m, cache.n)
    return β
end

function _fast_lanczos_axpy!(::Type{<:CPUEngine}, coeff, ketk, initial, m, n)
    myaxpy!(coeff, ketk, initial)
end

function _fast_lanczos_axpy!(::Type{<:GPUEngine}, coeff, ketk, initial, m, n)
    myaxpy!(coeff, [CuArray.(ketk[i, j]) for i in 1 : m, j in 1 : n], initial)
end
