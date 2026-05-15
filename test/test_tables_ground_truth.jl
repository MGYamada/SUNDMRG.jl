using JLD2
using SUNRepresentations: SUNIrrep

_ground_truth_irrep(::Val{Nc}, weight) where Nc = SUNIrrep{Nc}(Tuple(weight))
_ground_truth_key(::Val{Nc}, weights) where Nc = Tuple(_ground_truth_irrep(Val(Nc), weight) for weight in weights)

function _ground_truth_tables(Nc, widthmax)
    JLD2.load(joinpath(@__DIR__, "..", "jld2", "table_SU$(Nc)_$(widthmax).jld2"), "tables")
end

function _ground_truth_table5_reference(::Val{Nc}, key) where Nc
    funda = SUNDMRG.fundamentalirrep(Val(Nc))
    αj, βl, αi, γ, βk = key
    SUNDMRG._6ν(αj, funda, βl, αi, γ, βk)[1, :, 1, :] .* SUNDMRG._3ν(αi, funda, βk)[1, 1]
end

function _ground_truth_table4_reference(::Val{Nc}, key) where Nc
    adjoint = SUNDMRG.adjointirrep(Val(Nc))
    trivial = SUNDMRG.trivialirrep(Val(Nc))
    α1, β1, γ, α2, β2 = key
    SUNDMRG._9ν(α1, β1, γ, adjoint, adjoint, trivial, α2, β2, γ)[:, :, :, :, 1, 1]
end

@testset "SU(3)-SU(5) JLD2 table ground truth" begin
    cases = (
        (
            Nc = 3,
            widthmax = 13,
            three = ((1, 0, 0), (1, 1, 0), (0, 0, 0)),
            six = ((1, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 0), (1, 1, 0)),
            nine = ((1, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 0), (1, 0, 0)),
        ),
        (
            Nc = 4,
            widthmax = 9,
            three = ((1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0)),
            six = ((1, 1, 0, 0), (1, 1, 1, 0), (1, 0, 0, 0), (0, 0, 0, 0), (1, 1, 0, 0)),
            nine = ((1, 0, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)),
        ),
        (
            Nc = 5,
            widthmax = 3,
            three = ((1, 0, 0, 0, 0), (1, 0, 0, 0, 0), (1, 1, 0, 0, 0)),
            six = ((1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (1, 1, 1, 0, 0), (0, 0, 0, 0, 0), (1, 1, 1, 1, 0)),
            nine = ((1, 0, 0, 0, 0), (1, 0, 0, 0, 0), (1, 1, 0, 0, 0), (1, 0, 0, 0, 0), (1, 0, 0, 0, 0)),
        ),
    )

    for case in cases
        tables = _ground_truth_tables(case.Nc, case.widthmax)
        table4, table5, table6 = tables[4], tables[5], tables[6]

        three_key = _ground_truth_key(Val(case.Nc), case.three)
        @test haskey(table6, three_key)
        @test SUNDMRG._3ν(three_key...) ≈ table6[three_key]

        six_key = _ground_truth_key(Val(case.Nc), case.six)
        @test haskey(table5, six_key)
        @test _ground_truth_table5_reference(Val(case.Nc), six_key) ≈ table5[six_key]

        nine_key = _ground_truth_key(Val(case.Nc), case.nine)
        @test haskey(table4, nine_key)
        @test _ground_truth_table4_reference(Val(case.Nc), nine_key) ≈ table4[nine_key]
    end
end
