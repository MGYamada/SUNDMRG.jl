@testset "dcinit partition helper" begin
    @test SUNDMRG.dcinit(0, 4) == [0, 0, 0, 0, 0]

    # Edge case: fewer tasks than workers.
    dc_small = SUNDMRG.dcinit(3, 5)
    @test length(dc_small) == 6
    @test dc_small == [0, 0, 1, 1, 2, 3]
    @test diff(dc_small) == [0, 1, 0, 1, 1]

    # Divisibility case: perfectly even partition.
    dc_even = SUNDMRG.dcinit(12, 4)
    @test dc_even == [0, 3, 6, 9, 12]
    @test all(==(3), diff(dc_even))

    # General monotonic / coverage sanity.
    for (N, Ncpu) in ((1, 3), (7, 4), (13, 6), (64, 8))
        dc = SUNDMRG.dcinit(N, Ncpu)
        @test first(dc) == 0
        @test last(dc) == N
        @test issorted(dc)
        @test all(x -> x in (fld(N, Ncpu), cld(N, Ncpu)), diff(dc))
        @test maximum(diff(dc)) - minimum(diff(dc)) <= 1
    end
end

@testset "DMRG step bond helpers" begin
    bonds, x_conn, y_conn = SUNDMRG._step_bonds(1, 1, 2, :square)
    @test bonds == [(1, 1)]
    @test x_conn == 1
    @test y_conn == 1

    bonds, x_conn, y_conn = SUNDMRG._step_bonds(3, 3, 2, :square)
    @test bonds == [(1, 1), (2, 2), (2, 1)]
    @test x_conn == 2
    @test y_conn == 2

    bonds, x_conn, y_conn = SUNDMRG._step_bonds(1, 1, 2, :honeycombZC)
    @test isempty(bonds)
    @test x_conn == 1
    @test y_conn == 1

    @test SUNDMRG._rank_held_bonds([(1, 1), (2, 2), (2, 1)], 2, 0) == [(1, 1), (0, 0)]
    @test SUNDMRG._rank_held_bonds([(1, 1), (2, 2), (2, 1)], 2, 1) == [(2, 2), (2, 1)]
end

@testset "DMRG step allocation helpers" begin
    OM = [1 0; 2 1]
    Ψ = SUNDMRG._initial_step_wavefunction(nothing, [2, 3], [4, 5], OM, 2, 0, SUNDMRG.CPUEngine)
    @test size(Ψ) == (2, 2)
    @test length(Ψ[1, 1]) == 1
    @test length(Ψ[1, 2]) == 0
    @test length(Ψ[2, 1]) == 0
    @test length(Ψ[2, 2]) == 1
    @test size(Ψ[1, 1][1]) == (2, 4)
    @test size(Ψ[2, 2][1]) == (3, 5)

    guessed = [[rand(1, 1)]]
    copied = SUNDMRG._initial_step_wavefunction(guessed, [1], [1], [1;;], 1, 0, SUNDMRG.CPUEngine)
    @test copied == guessed
    @test copied !== guessed
    @test copied[1, 1] !== guessed[1, 1]

    newblock, newblock_enl, trmat, newtensor_dict = SUNDMRG._step_result_buffers(Val(2), SUNDMRG.CPUEngine)
    @test newblock isa Vector{SUNDMRG.Block{2}}
    @test newblock_enl isa Vector{SUNDMRG.EnlargedBlockCPU{2}}
    @test trmat isa Vector{Vector{Matrix{Float64}}}
    @test newtensor_dict isa Vector{Dict{Int64, Matrix{Vector{Matrix{Float64}}}}}
end

@testset "DMRG step side context helper" begin
    sys = :sys_block
    env = :env_block
    sys_enl = :sys_enlarged
    env_enl = :env_enlarged
    sys_dp = [:sys_dp]
    env_dp = [:env_dp]

    side = SUNDMRG._step_side_context(1, :l, 2, 3, [10, 20], [30, 40, 50], [:a, :b], [:c, :d, :e], sys_dp, env_dp, 4, 5, sys, env, sys_enl, env_enl)
    @test side isa SUNDMRG._StepSideContext
    @test side.len == 2
    @test side.ms == [10, 20]
    @test side.betas == [:a, :b]
    @test side.dp === sys_dp
    @test side.conn == 4
    @test side.label == :l
    @test side.block == sys
    @test side.block_enl == sys_enl

    side = SUNDMRG._step_side_context(2, :l, 2, 3, [10, 20], [30, 40, 50], [:a, :b], [:c, :d, :e], sys_dp, env_dp, 4, 5, sys, env, sys_enl, env_enl)
    @test side isa SUNDMRG._StepSideContext
    @test side.len == 3
    @test side.ms == [30, 40, 50]
    @test side.betas == [:c, :d, :e]
    @test side.dp === env_dp
    @test side.conn == 5
    @test side.label == :r
    @test side.block == env
    @test side.block_enl == env_enl
end

@testset "Density matrix balancer helper" begin
    @test SUNDMRG._density_matrix_balancer(Int[], 1, 0) == Int[]
    @test SUNDMRG._density_matrix_balancer([10, 20, 30], 1, 0) == [0, 0, 0]
    @test SUNDMRG._density_matrix_balancer([10, 20, 30], 4, 1) == [0, 0, 0]

    balancer = SUNDMRG._density_matrix_balancer([10, 20, 30], 3, 0)
    @test length(balancer) == 3
    @test all(x -> 0 <= x < 3, balancer)

    ρs = SUNDMRG._empty_density_matrix_vector(SUNDMRG.CPUEngine)
    @test ρs isa Vector{Matrix{Float64}}
    @test isempty(ρs)

    density_context = SUNDMRG._StepDensityContext(nothing, 1, 1, SUNDMRG.CPUEngine, 0.0, 1, -Inf, false)
    eigenpairs, elapsed = SUNDMRG._density_eigendecomposition([[2.0 0.0; 0.0 3.0]], density_context)
    @test elapsed >= 0.0
    @test length(eigenpairs) == 1
    @test eigenpairs[1][1] ≈ [2.0, 3.0]

    empty_side = SUNDMRG._StepSideContext(0, Int[], [], [], 1, :l, nothing, nothing)
    ee, esi, transformation_matrix, msnew, Hnew, indices = SUNDMRG._density_truncation_basis([], [], empty_side, [], nothing, nothing, density_context, Val(2), 1)
    @test ee == 0.0
    @test isempty(esi)
    @test length(transformation_matrix) == 1
    @test size(transformation_matrix[1]) == (0, 0)
    @test isempty(msnew)
    @test isempty(Hnew)
    @test isempty(indices)
end
