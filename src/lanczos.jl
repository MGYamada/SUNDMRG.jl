const tol_wavefunction = 1e-13
const tol_Lanczos = 1e-13

MyMatrix = Union{Matrix{Vector{Matrix{Float64}}}, Matrix{Vector{CuMatrix{Float64}}}}

function LinearAlgebra.dot(x::MyMatrix, y::MyMatrix)
    s = 0.0
    for (IX, IY) in zip(eachindex(x), eachindex(y))
        for (JX, JY) in zip(eachindex(x[IX]), eachindex(y[IY]))
            s += dot(x[IX][JX], y[IY][JY])
        end
    end
    s
end

function LinearAlgebra.axpy!(α, x::MyMatrix, y::MyMatrix)
    for (IY, IX) in zip(eachindex(y), eachindex(x))
        for (JY, JX) in zip(eachindex(y[IY]), eachindex(x[IX]))
            @. y[IY][JY] += α * x[IX][JX]
        end
    end
    y
end

function LinearAlgebra.axpby!(α, x::MyMatrix, β, y::MyMatrix)
    for (IX, IY) in zip(eachindex(x), eachindex(y))
        for (JX, JY) in zip(eachindex(x[IX]), eachindex(y[IY]))
            @. y[IY][JY] = α * x[IX][JX] + β * y[IY][JY]
        end
    end
    y
end

function LinearAlgebra.rmul!(A::MyMatrix, b::Number)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] .*= b
        end
    end
    A
end

function LinearAlgebra.rdiv!(A::MyMatrix, b::Number)
    for I in eachindex(A)
        for J in eachindex(A[I])
            A[I][J] ./= b
        end
    end
    A
end

function Base.copyto!(dest::MyMatrix, src::MyMatrix)
    for (IX, IY) in zip(eachindex(dest), eachindex(src))
        for (JX, JY) in zip(eachindex(dest[IX]), eachindex(src[IY]))
            dest[IX][JX] .= src[IY][JY]
        end
    end
    dest
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
        axpby!(1.0 + valshift, x, -1.0, r)
        copyto!(p, r)
        normold = MPI.Allreduce(dot(r, r), MPI.SUM, comm)
        j = 1
        while true
            A!(Ap, p)
            axpy!(-valshift, p, Ap)
            α = normold / MPI.Allreduce(dot(p, Ap), MPI.SUM, comm)
            axpy!(α, p, x)
            axpy!(-α, Ap, r)
            normnew = MPI.Allreduce(dot(r, r), MPI.SUM, comm)
            if j == 10 || normnew < 1e-8
                break
            end
            β = normnew / normold
            axpby!(1.0, r, β, p)
            normold = normnew
            j += 1
        end
        rdiv!(x, sqrt(MPI.Allreduce(dot(x, x), MPI.SUM, comm)))
        A!(r, x)
        valnew = MPI.Allreduce(dot(x, r), MPI.SUM, comm)
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
    ketkm1 = deepcopy(initial)
    for I in eachindex(ketkm1)
        for J in eachindex(ketkm1[I])
            ketkm1[I][J] .= 0.0
        end
    end
    rdiv!(initial, sqrt(MPI.Allreduce(dot(initial, initial), MPI.SUM, comm)))
    ketk = deepcopy(initial)
    ketk1 = deepcopy(ketk)
    β = 0.0
    αlist = Float64[]
    βlist = Float64[]
    vals = Float64[]
    vecs = zeros(0, 0)
    vold = Inf
    if alg == :fast
        m, n = size(ketk)
        ketk_list = Matrix{Vector{Matrix{Float64}}}[]
    end
    k = 1
    while true
        A!(ketk1, ketk)
        α = MPI.Allreduce(dot(ketk, ketk1), MPI.SUM, comm)
        push!(αlist, α)
        if k >= position
            if rank == 0
                vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
            end
            vals = MPI.bcast(vals, 0, comm)::Vector{Float64}
            if k == maxiter || abs((vals[position] - vold) / vals[position]) < tol_Lanczos
                break
            end
            vold = vals[position]
        end
        axpy!(-β, ketkm1, ketk1)
        axpy!(-α, ketk, ketk1)
        β = sqrt(MPI.Allreduce(dot(ketk1, ketk1), MPI.SUM, comm))
        if β == 0.0
            if rank == 0
                vals, vecs = LAPACK.stev!('V', copy(αlist), copy(βlist))
            end
            vals = MPI.bcast(vals, 0, comm)::Vector{Float64}
            break
        end
        rdiv!(ketk1, β)
        copyto!(ketkm1, ketk)
        copyto!(ketk, ketk1)
        push!(βlist, β)
        if alg == :fast
            push!(ketk_list, map(x -> Array.(x), ketk))
        end
        k += 1
    end
    vecs = MPI.bcast(vecs, 0, comm)::Matrix{Float64}

    for I in eachindex(ketkm1)
        for J in eachindex(ketkm1[I])
            ketkm1[I][J] .= 0.0
        end
    end
    copyto!(ketk, initial)
    β = 0.0
    position = min(position, size(vecs, 2))
    rmul!(initial, vecs[1, position])
    for k in 1 : size(vecs, 1) - 1
        if alg == :slow
            A!(ketk1, ketk)
            α = αlist[k]
            axpy!(-β, ketkm1, ketk1)
            axpy!(-α, ketk, ketk1)
            β = βlist[k]
            rdiv!(ketk1, β)
            copyto!(ketkm1, ketk)
            copyto!(ketk, ketk1)
            axpy!(vecs[k + 1, position], ketk, initial)
        else
            if engine <: GPUEngine
                axpy!(vecs[k + 1, position], [CuArray.(ketk_list[k][i, j]) for i in 1 : m, j in 1 : n], initial)
            else
                axpy!(vecs[k + 1, position], ketk_list[k], initial)
            end
        end
    end
    rdiv!(initial, sqrt(MPI.Allreduce(dot(initial, initial), MPI.SUM, comm)))
    A!(ketk, initial)
    val = MPI.Allreduce(dot(initial, ketk), MPI.SUM, comm)
    val2 = val ^ 2
    var = MPI.Allreduce(dot(ketk, ketk), MPI.SUM, comm)
    if abs((val - vals[position]) / val) < tol_wavefunction && abs((var - val2) / val2) < tol_wavefunction
        rtnval = val
    else
        rtnval = CG!(A!, val, initial, ketk, ketkm1, ketk1, comm, rank)
    end
    rtnval
end