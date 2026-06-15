using LinearAlgebra: I, dot
using SUNRepresentations: directproduct, dim, weight

function _test_young_rows(Nc, widthmax)
    rows = NTuple{Nc, Int}[]

    function extend!(prefix, max_row)
        if length(prefix) == Nc - 1
            push!(rows, Tuple(vcat(prefix, 0)))
            return nothing
        end
        for row in 0 : max_row
            extend!(vcat(prefix, row), row)
        end
        return nothing
    end

    for first_row in 0 : widthmax
        extend!([first_row], first_row)
    end

    rows
end

function _test_hook_tableau_count(rows)
    rows = filter(>(0), collect(rows))
    isempty(rows) && return Int128(1)

    hooks = Int[]
    for i in eachindex(rows), j in 1 : rows[i]
        below = count(k -> rows[k] >= j, i + 1 : length(rows))
        push!(hooks, rows[i] - j + below + 1)
    end

    Int128(factorial(big(sum(rows))) ÷ prod(big.(hooks)))
end

function _test_weyl_dimension(rows)
    λ = collect(rows)
    Nc = length(λ)
    result = 1 // 1
    for i in 1 : Nc - 1, j in i + 1 : Nc
        result *= (λ[i] - λ[j] + j - i) // (j - i)
    end
    @assert denominator(result) == 1
    numerator(result)
end

function _test_normalized_product_weight(ν1, ν2, ν)
    Nc = length(weight(ν1))
    μ1, μ2, μ = map(collect ∘ weight, (ν1, ν2, ν))
    f1, f2 = map(sum, (μ1, μ2))
    μ .+= (f1 + f2 - sum(μ)) ÷ Nc
    map(Tuple, (μ1, μ2, μ))
end

@testset "Irrep enumeration follows Young row-length definitions" begin
    RT = SUNDMRG.RepresentationTheory

    for Nc in 2 : 5, widthmax in 0 : 3
        actual = weight.(RT.irreplist(Nc, widthmax))
        expected = _test_young_rows(Nc, widthmax)
        @test actual == expected
        @test length(actual) == binomial(widthmax + Nc - 1, Nc - 1)
    end
end

@testset "Young diagram multiplicities follow the hook-length formula" begin
    RT = SUNDMRG.RepresentationTheory

    for rows in (
        Int[],
        [1],
        [2],
        [1, 1],
        [2, 1],
        [3, 1],
        [3, 2],
        [2, 1, 1],
        [3, 2, 1],
    )
        @test RT.multiplicity(rows) == _test_hook_tableau_count(rows)

        if !isempty(rows)
            V, E, _ = RT.SYTdiagram(rows)
            B, _ = RT.bf(sum(rows), V, E, Int128)
            @test B[1][1] == RT.multiplicity(rows)
        end
    end
end

@testset "SUNIrrep dimensions follow the Weyl dimension formula" begin
    RT = SUNDMRG.RepresentationTheory

    for Nc in 2 : 5
        for irrep in RT.irreplist(Nc, 3)
            @test dim(irrep) == _test_weyl_dimension(weight(irrep))
        end

        @test dim(RT.trivialirrep(Val(Nc))) == 1
        @test dim(RT.fundamentalirrep(Val(Nc))) == Nc
        @test dim(RT.adjointirrep(Val(Nc))) == Nc ^ 2 - 1
    end
end

@testset "Tensor-product multiplicities preserve dimensions" begin
    RT = SUNDMRG.RepresentationTheory

    for Nc in 2 : 5
        trivial = RT.trivialirrep(Val(Nc))
        funda = RT.fundamentalirrep(Val(Nc))
        adjoint = RT.adjointirrep(Val(Nc))
        probes = (trivial, funda, adjoint)

        for α in probes, β in probes
            decomposition = directproduct(α, β)
            @test sum(mult * dim(γ) for (γ, mult) in decomposition) == dim(α) * dim(β)
            for (γ, mult) in decomposition
                @test RT.outer_multiplicity(α, β, γ) == mult
            end
        end

        OM = RT.OM_matrix(collect(probes), collect(probes), trivial)
        @test OM == [RT.outer_multiplicity(α, β, trivial) for α in probes, β in probes]
    end
end

@testset "Subduction coefficients form orthonormal outer-multiplicity bases" begin
    RT = SUNDMRG.RepresentationTheory

    adjoint = RT.adjointirrep(Val(3))
    ν1, ν2, ν = _test_normalized_product_weight(adjoint, adjoint, adjoint)
    vec1 = RT.sparsevec2(Int128[1], [1.0], RT.multiplicity(ν1))
    vec2 = RT.sparsevec2(Int128[1], [1.0], RT.multiplicity(ν2))

    coeffs = RT.SDC(ν1, ν2, ν, vec1, vec2)
    @test length(coeffs) == RT.outer_multiplicity(adjoint, adjoint, adjoint)
    @test [dot(coeffs[i], coeffs[j]) for i in eachindex(coeffs), j in eachindex(coeffs)] ≈ Matrix(I, length(coeffs), length(coeffs))
end
