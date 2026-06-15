using LinearAlgebra: dot
using SUNRepresentations: weight

function _test_one_sparsevec(ν)
    SUNDMRG.sparsevec2(Int128[1], [1.0], SUNDMRG.multiplicity(ν))
end

function _test_weight_tuples_3(ν1, ν2, ν)
    Nc = length(weight(ν1))
    μ1, μ2, μ = map(collect ∘ weight, (ν1, ν2, ν))
    f1, f2 = map(sum, (μ1, μ2))
    μ .+= (f1 + f2 - sum(μ)) ÷ Nc
    map(Tuple, (μ1, μ2, μ))
end

function _test_weight_tuples_6(ν1, ν2, ν12, ν3, ν, ν23)
    Nc = length(weight(ν1))
    μ1, μ2, μ12, μ3, μ, μ23 = map(collect ∘ weight, (ν1, ν2, ν12, ν3, ν, ν23))
    f1, f2, f3 = map(sum, (μ1, μ2, μ3))
    μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
    μ23 .+= (f2 + f3 - sum(μ23)) ÷ Nc
    μ .+= (sum(μ12) + f3 - sum(μ)) ÷ Nc
    map(Tuple, (μ1, μ2, μ12, μ3, μ, μ23))
end

function _test_weight_tuples_9(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)
    Nc = length(weight(ν1))
    μ1, μ2, μ12, μ3, μ4, μ34, μ13, μ24, μ = map(collect ∘ weight, (ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν))
    f1, f2, f3, f4 = map(sum, (μ1, μ2, μ3, μ4))
    μ12 .+= (f1 + f2 - sum(μ12)) ÷ Nc
    μ34 .+= (f3 + f4 - sum(μ34)) ÷ Nc
    μ13 .+= (f1 + f3 - sum(μ13)) ÷ Nc
    μ24 .+= (f2 + f4 - sum(μ24)) ÷ Nc
    μ .+= (sum(μ12) + sum(μ34) - sum(μ)) ÷ Nc
    map(Tuple, (μ1, μ2, μ12, μ3, μ4, μ34, μ13, μ24, μ))
end

function _test_3ν_from_sdc(ν1, ν2, ν)
    ν1, ν2, ν = _test_weight_tuples_3(ν1, ν2, ν)
    vec1 = _test_one_sparsevec(ν1)
    vec2 = _test_one_sparsevec(ν2)
    right = SUNDMRG.SDC(ν1, ν2, ν, vec1, vec2; perm = true)
    left = SUNDMRG.SDC(ν2, ν1, ν, vec1, vec2)
    [dot(left[τ′], right[τ]) for τ′ in eachindex(left), τ in eachindex(right)]
end

function _test_6ν_from_sdc(ν1, ν2, ν12, ν3, ν, ν23)
    ν1, ν2, ν12, ν3, ν, ν23 = _test_weight_tuples_6(ν1, ν2, ν12, ν3, ν, ν23)
    vec1 = _test_one_sparsevec(ν1)
    vec2 = _test_one_sparsevec(ν2)
    vec3 = _test_one_sparsevec(ν3)
    vec12 = SUNDMRG.SDC(ν1, ν2, ν12, vec1, vec2)
    right = [SUNDMRG.SDC(ν12, ν3, ν, vec12[τ12], vec3) for τ12 in eachindex(vec12)]
    vec23 = SUNDMRG.SDC(ν2, ν3, ν23, vec2, vec3)
    left = [SUNDMRG.SDC(ν1, ν23, ν, vec1, vec23[τ23]) for τ23 in eachindex(vec23)]
    [dot(left[τ23][τ′], right[τ12][τ]) for τ23 in eachindex(vec23), τ′ in eachindex(left[1]), τ12 in eachindex(vec12), τ in eachindex(right[1])]
end

function _test_9ν_from_sdc(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν; perm = true)
    ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν = _test_weight_tuples_9(ν1, ν2, ν12, ν3, ν4, ν34, ν13, ν24, ν)
    N1 = isempty(ν1) ? 0 : sum(ν1)
    N4 = isempty(ν4) ? 0 : sum(ν4)
    vec1 = _test_one_sparsevec(ν1)
    vec2 = _test_one_sparsevec(ν2)
    vec3 = _test_one_sparsevec(ν3)
    vec4 = _test_one_sparsevec(ν4)
    vec12 = SUNDMRG.SDC(ν1, ν2, ν12, vec1, vec2)
    vec34 = SUNDMRG.SDC(ν3, ν4, ν34, vec3, vec4)
    right = [SUNDMRG.SDC(ν12, ν34, ν, vec12[τ12], vec34[τ34]; perm = perm, f1 = N1, f4 = N4) for τ12 in eachindex(vec12), τ34 in eachindex(vec34)]
    vec13 = SUNDMRG.SDC(ν1, ν3, ν13, vec1, vec3)
    vec24 = SUNDMRG.SDC(ν2, ν4, ν24, vec2, vec4)
    left = [SUNDMRG.SDC(ν13, ν24, ν, vec13[τ13], vec24[τ24]) for τ13 in eachindex(vec13), τ24 in eachindex(vec24)]
    [dot(left[τ13, τ24][τ′], right[τ12, τ34][τ]) for τ13 in eachindex(vec13), τ24 in eachindex(vec24), τ′ in eachindex(left[1, 1]), τ12 in eachindex(vec12), τ34 in eachindex(vec34), τ in eachindex(right[1, 1])]
end

@testset "Wigner/Racah coefficients follow SDC recoupling definitions" begin
    trivial = SUNDMRG.trivialirrep(Val(2))
    funda = SUNDMRG.fundamentalirrep(Val(2))
    adjoint = SUNDMRG.adjointirrep(Val(2))

    @test SUNDMRG._3ν(funda, funda, adjoint) ≈ _test_3ν_from_sdc(funda, funda, adjoint)
    @test SUNDMRG._6ν(funda, funda, adjoint, funda, funda, adjoint) ≈
          _test_6ν_from_sdc(funda, funda, adjoint, funda, funda, adjoint)

    nineν_case = (trivial, funda, funda, adjoint, trivial, adjoint, adjoint, funda, funda)
    with_p23 = _test_9ν_from_sdc(nineν_case...)
    without_p23 = _test_9ν_from_sdc(nineν_case...; perm = false)

    @test SUNDMRG._9ν(nineν_case...) ≈ with_p23
    @test !(SUNDMRG._9ν(nineν_case...) ≈ without_p23)
    @test only(without_p23) ≈ 0.5
end
