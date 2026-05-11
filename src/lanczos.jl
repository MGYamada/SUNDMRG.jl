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
        if abs((valnew - val) / valnew) < tol_wavefunction
            break
        end
        val = valnew
    end
    valnew
end

"""
Lanczos!(A!, initial, position, comm, rank, engine; maxiter = 100, alg = :slow)
returns eigenpairs of a linear map A!
"""
function Lanczos!(A!::Function, initial, position, comm, rank, engine; maxiter = 100, alg = :slow)
    Lanczos!(A!, initial, position, comm, rank, engine, _alg_value(alg); maxiter = maxiter)
end

_alg_value(alg::Symbol) = Val(alg)
_alg_value(alg::Val) = alg

function Lanczos!(A!::Function, initial, position, comm, rank, engine, mode::Val; maxiter = 100)
    ketkm1 = _zero_lanczos_vector(initial)
    myrdiv!(initial, sqrt(MPI.Allreduce(mydot(initial, initial), MPI.SUM, comm)))
    ketk = deepcopy(initial)
    ketk1 = deepcopy(ketk)
    β = 0.0
    αlist = Float64[]
    βlist = Float64[]
    vals = Float64[]
    vecs = zeros(0, 0)
    vold = Inf
    cache = _init_lanczos_cache(mode, ketk)
    k = 1
    while true
        myzero!(ketk1)
        A!(ketk1, ketk)
        α = MPI.Allreduce(mydot(ketk, ketk1), MPI.SUM, comm)
        push!(αlist, α)
        if k >= position
            vals, vecs = _lanczos_ritz_vectors(αlist, βlist, comm, rank)
            if k == maxiter || _lanczos_eigenvalue_converged(vals[position], vold)
                break
            end
            vold = vals[position]
        end
        myaxpy!(-β, ketkm1, ketk1)
        myaxpy!(-α, ketk, ketk1)
        β = sqrt(MPI.Allreduce(mydot(ketk1, ketk1), MPI.SUM, comm))
        if β == 0.0
            vals, vecs = _lanczos_ritz_vectors(αlist, βlist, comm, rank)
            break
        end
        myrdiv!(ketk1, β)
        mycopyto!(ketkm1, ketk)
        mycopyto!(ketk, ketk1)
        push!(βlist, β)
        _cache_lanczos_vector!(mode, cache, ketk)
        k += 1
    end
    vecs = MPI.bcast(vecs, 0, comm)::Matrix{Float64}
    if size(vecs, 2) == 0
        return first(αlist)
    end

    position = _reconstruct_lanczos_vector!(mode, A!, engine, initial, ketk, ketk1, ketkm1, vecs, αlist, βlist, position, cache)
    _refine_lanczos_vector!(A!, initial, ketk, ketkm1, ketk1, vals[position], comm, rank)
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

_lanczos_eigenvalue_converged(val, vold) = abs(val - vold) < tol_Lanczos * max(abs(val), 1.0)
_lanczos_wavefunction_converged(val, target_val, var) = abs(val - target_val) < tol_wavefunction * max(abs(val), 1.0) && abs(var - val ^ 2) < tol_wavefunction * max(abs(val ^ 2), 1.0)

function _reconstruct_lanczos_vector!(mode::Val, A!, engine, initial, ketk, ketk1, ketkm1, vecs, αlist, βlist, position, cache)
    myzero!(ketkm1)
    mycopyto!(ketk, initial)
    β = 0.0
    position = min(position, size(vecs, 2))
    myrmul!(initial, vecs[1, position])
    for k in 1 : size(vecs, 1) - 1
        β = _lanczos_reconstruct_step!(mode, A!, engine, vecs[k + 1, position], cache, initial, ketk, ketk1, ketkm1, αlist, βlist, k, β)
    end
    return position
end

function _refine_lanczos_vector!(A!, initial, ketk, ketkm1, ketk1, target_val, comm, rank)
    myrdiv!(initial, sqrt(MPI.Allreduce(mydot(initial, initial), MPI.SUM, comm)))
    myzero!(ketk)
    A!(ketk, initial)
    val = MPI.Allreduce(mydot(initial, ketk), MPI.SUM, comm)
    var = MPI.Allreduce(mydot(ketk, ketk), MPI.SUM, comm)
    if _lanczos_wavefunction_converged(val, target_val, var)
        return val
    end
    return CG!(A!, val, initial, ketk, ketkm1, ketk1, comm, rank)
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
