import Base: +, *, getindex, convert
import LinearAlgebra: norm, dot, lmul!

const tol_subduction = 1e-13

struct SparseVector2{Tv, Ti <: Integer}
    n::Ti
    nzind::Vector{Ti}
    nzval::Vector{Tv}
end

spzeros2(::Type{Tv}, ::Type{Ti}, len::Integer) where {Tv, Ti<:Integer} = SparseVector2(len, Ti[], Tv[])

function sparsevec2(I::AbstractVector{<:Integer}, V::Vector, len::Integer)
    if !isempty(I)
        p = _sortperm(I)
        permute!(I, p)
        permute!(V, p)
    end
    SparseVector2(len, I, V)
end

function _sortperm(v; order::Base.Ordering = Base.Forward)
    p = collect(eachindex(v))
    ThreadsX.sort!(p; order = Base.Perm(order, v))
end

function getindex(x::SparseVector2{Tv, Ti}, i::Integer) where {Tv, Ti}
    # checkbounds(x, i)
    ii = searchsortedfirst(x.nzind, convert(Ti, i))
    (ii <= length(x.nzind) && x.nzind[ii] == i) ? x.nzval[ii] : zero(Tv)
end

function +(x::SparseVector2{Tv, Ti}, y::SparseVector2{Tv, Ti}) where {Tv, Ti}
    _plus(x, y, eachindex(x.nzind), eachindex(y.nzind), 1)
end

function _plus(x::SparseVector2{Tv, Ti}, y::SparseVector2{Tv, Ti}, xinds, yinds, depth) where {Tv, Ti}
    mx = length(xinds)
    my = length(yinds)
    if mx == 0
        return SparseVector2(y.n, y.nzind[yinds], y.nzval[yinds])
    elseif my == 0
        return SparseVector2(x.n, x.nzind[xinds], x.nzval[xinds])
    elseif depth >= 7 || (mx < 100 && my < 100)
        ir = 0
        ix = 1
        iy = 1
        cap = mx + my
        rind = Vector{Ti}(undef, cap)
        rval = Vector{Tv}(undef, cap)
        @inbounds while ix <= mx && iy <= my
            jx = x.nzind[xinds[ix]]
            jy = y.nzind[yinds[iy]]
            if jx == jy
                v = x.nzval[xinds[ix]] + y.nzval[yinds[iy]]
                if abs(v) > tol_subduction
                    ir += 1
                    rind[ir] = jx
                    rval[ir] = v
                end
                ix += 1
                iy += 1
            elseif jx < jy
                v = x.nzval[xinds[ix]]
                if abs(v) > tol_subduction
                    ir += 1
                    rind[ir] = jx
                    rval[ir] = v
                end
                ix += 1
            else
                v = y.nzval[yinds[iy]]
                if abs(v) > tol_subduction
                    ir += 1
                    rind[ir] = jy
                    rval[ir] = v
                end
                iy += 1
            end
        end
        @inbounds while ix <= mx
            v = x.nzval[xinds[ix]]
            if abs(v) > tol_subduction
                ir += 1
                rind[ir] = x.nzind[xinds[ix]]
                rval[ir] = v
            end
            ix += 1
        end
        @inbounds while iy <= my
            v = y.nzval[yinds[iy]]
            if abs(v) > tol_subduction
                ir += 1
                rind[ir] = y.nzind[yinds[iy]]
                rval[ir] = v
            end
            iy += 1
        end
        resize!(rind, ir)
        resize!(rval, ir)
        return SparseVector2(x.n, rind, rval)
    end
    pivot = (min(x.nzind[xinds[1]], y.nzind[yinds[1]]) + max(x.nzind[xinds[end]], y.nzind[yinds[end]])) >> 1
    px = @views searchsortedlast(x.nzind[xinds], pivot)
    py = @views searchsortedlast(y.nzind[yinds], pivot)
    left = Threads.@spawn _plus(x, y, xinds[1 : px], yinds[1 : py], depth + 1)
    right = Threads.@spawn _plus(x, y, xinds[(px + 1) : end], yinds[(py + 1) : end], depth + 1)
    fleft = fetch(left)
    fright = fetch(right)
    append!(fleft.nzind, fright.nzind)
    append!(fleft.nzval, fright.nzval)
    fleft
end

(*)(x::SparseVector2{Tv, Ti}, a::Number) where {Tv, Ti} = SparseVector2(x.n, copy(x.nzind), x.nzval .* a)
(*)(a::Number, x::SparseVector2{Tv, Ti}) where {Tv, Ti} = SparseVector2(x.n, copy(x.nzind), a .* x.nzval)

function lmul!(a::Real, x::SparseVector2)
    rmul!(x.nzval, a)
    x
end

norm(x::SparseVector2, p::Real = 2) = norm(x.nzval, p)

function dot(x::SparseVector2, y::SparseVector2)
    xj = 1
    xj_last = length(x.nzind)
    yj = 1
    yj_last = length(y.nzind)
    s = dot(zero(eltype(x.nzval)), zero(eltype(y.nzval)))
    @inbounds while xj <= xj_last && yj <= yj_last
        ix = x.nzind[xj]
        iy = y.nzind[yj]
        if ix == iy
            s += dot(x.nzval[xj], y.nzval[yj])
            xj += 1
            yj += 1
        elseif ix < iy
            xj += 1
        else
            yj += 1
        end
    end
    s
end

# function droptol!(x::SparseVector2, tol)
#     x_writepos = 1
#     @inbounds for xk in 1 : length(x.nzind)
#         xi = x.nzind[xk]
#         xv = x.nzval[xk]
#         if abs(xv) > tol
#             if x_writepos != xk
#                 x.nzind[x_writepos] = xi
#                 x.nzval[x_writepos] = xv
#             end
#             x_writepos += 1
#         end
#     end
#     x_nnz = x_writepos - 1
#     resize!(x.nzval, x_nnz)
#     resize!(x.nzind, x_nnz)
#     x
# end