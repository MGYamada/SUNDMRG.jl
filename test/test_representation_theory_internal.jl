using LinearAlgebra: diag
using SUNRepresentations: directproduct, dim, weight

@testset "RepresentationTheory submodule access" begin
    RT = SUNDMRG.RepresentationTheory

    @test RT.RepresentationTable == NTuple{6, Any}
    @test RT.irrep(3, 3) == SUNDMRG.irrep(3, 3)
    @test RT.trivialirrep(Val(3)) == SUNDMRG.trivialirrep(Val(3))
    @test RT.fundamentalirrep(Val(3)) == SUNDMRG.fundamentalirrep(Val(3))
    @test RT.adjointirrep(Val(3)) == SUNDMRG.adjointirrep(Val(3))
end

@testset "RepresentationTheory irrep enumeration" begin
    RT = SUNDMRG.RepresentationTheory

    irreps = RT.irreplist(3, 2)
    @test weight.(irreps) == [
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (2, 0, 0),
        (2, 1, 0),
        (2, 2, 0),
    ]
    @test dim.(irreps) == [1, 3, 3, 6, 8, 6]
    @test all(weight(irrep)[1] <= 2 for irrep in irreps)

    trivial = RT.trivialirrep(Val(3))
    funda = RT.fundamentalirrep(Val(3))
    adjoint = RT.adjointirrep(Val(3))
    @test RT.outer_multiplicity(funda, funda, adjoint) == 0
    @test RT.outer_multiplicity(funda, funda, irreps[4]) == 1
    @test RT.outer_multiplicity(funda, trivial, funda) == 1
    @test RT.OM_matrix(irreps[1:3], trivial) == RT.OM_matrix(irreps[1:3], irreps[1:3], trivial)
    @test haskey(directproduct(funda, trivial), funda)
end

@testset "Young diagram and SYT helpers" begin
    RT = SUNDMRG.RepresentationTheory

    @test RT.multiplicity(Int[]) == 1
    @test RT.multiplicity([1]) == 1
    @test RT.multiplicity([2, 1]) == 2
    @test RT.multiplicity([3, 2]) == 5

    V, E, D = RT.SYTdiagram([2, 1])
    @test V == [[[1]], [[1, 1], [2]], [[2, 1]]]
    @test E == [[[1, 2]], [[1], [1]]]
    @test D == [[[3, 2]], [[2], [3]]]

    B, F = RT.bf(3, V, E, Int)
    @test B == [[2], [1, 1], [1]]
    @test F == [[[0, 1]], [[0], [0]]]
    @test RT.subdiagram([[1]], [2, 1], 1, E, D, B, F, Int) == (1, 2, 1)

    reps = RT.representatives([Int[]], [[1]], [1])
    @test reps == [(1,)]
end

@testset "Subduction coefficients" begin
    RT = SUNDMRG.RepresentationTheory

    onevec = RT.sparsevec2(Int128[1], [1.0], Int128(1))
    for key in (
        ((0, 0), (1, 0), (1, 0)),
        ((1, 0), (0, 0), (1, 0)),
        ((1, 0), (1, 0), (2, 0)),
        ((1, 0), (1, 0), (1, 1)),
    )
        coeffs = RT.SDC(key..., onevec, onevec)
        @test length(coeffs) == 1
        @test only(coeffs).n == 1
        @test only(coeffs).nzind == Int128[1]
        @test only(coeffs).nzval ≈ [1.0]
    end

    V, E, D = RT.SYTdiagram([2])
    B, F = RT.bf(2, V, E, Int)
    basis, dim_subspace, initj = RT._SDC((1,), (1,), (2,), E, D, B, F, Int)
    @test dim_subspace == 1
    @test initj == 1
    @test length(basis) == 1
    @test only(basis).nzind == [1]
    @test only(basis).nzval ≈ [1.0]
end

@testset "Wigner and Racah wrappers" begin
    RT = SUNDMRG.RepresentationTheory
    trivial = RT.trivialirrep(Val(2))
    funda = RT.fundamentalirrep(Val(2))
    adjoint = RT.adjointirrep(Val(2))

    @test RT.racahU(funda, funda, funda, funda, adjoint, adjoint) ≈
          RT.wigner6ν(funda, funda, adjoint, funda, funda, adjoint)

    w6 = RT.wigner6ν(funda, funda, adjoint, funda, funda, adjoint)
    @test RT.wigner6νrev(funda, funda, adjoint, funda, funda, adjoint) ≈ permutedims(w6, (3, 4, 1, 2))
    @test RT.wigner9j(1 // 2, 1 // 2, 0, 1 // 2, 1 // 2, 0, 0, 0, 0) ≈ 0.5

    impossible = RT.wigner6ν(trivial, trivial, funda, trivial, trivial, trivial)
    @test size(impossible) == (1, 1, 0, 0)
end

@testset "Subduction linear algebra helpers" begin
    RT = SUNDMRG.RepresentationTheory

    @test RT.findabsmax([0.0, -3.0, 2.0]) == (3.0, 2)
    @test_throws ArgumentError RT.findabsmax(Float64[])

    Q, R = RT.qrpos!([1.0 0.0; 0.0 -2.0])
    @test Q * R ≈ [1.0 0.0; 0.0 -2.0]
    @test diag(R) == [1.0, 2.0]
end

@testset "Representation table MPI lifecycle helpers" begin
    RT = SUNDMRG.RepresentationTheory

    SUNDMRG.MPI.Initialized() || SUNDMRG.MPI.Init(; threadlevel = SUNDMRG.MPI.THREAD_FUNNELED)

    @test RT._init_table_mpi!(false) == false
    @test RT._init_table_mpi!(true) == false
    @test RT._finalize_table_mpi!(false) == false
end
