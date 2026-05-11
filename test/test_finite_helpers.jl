@testset "Finite internal storage helpers" begin
    function sample_finite_block()
        trivial = SUNDMRG.trivialirrep(Val(2))
        SUNDMRG.Block(
            2,
            [(1, 2)],
            [trivial],
            [1],
            [1],
            Dict{Symbol, Vector{Matrix{Float64}}}(:H => [[2.0 0.0; 0.0 3.0]]),
        )
    end

    block = sample_finite_block()
    trmat = [[1.0 0.0; 0.0 1.0], reshape([1.0, 0.0], 2, 1)]

    @testset "memory storage" begin
        block_table = Dict{Tuple{Symbol, Int}, Any}()
        trmat_table = Dict{Tuple{Symbol, Int}, Any}()
        tensor_table = Dict{Tuple{Symbol, Int, Int}, Any}()
        storage = SUNDMRG.MemoryInternalStorage(block_table, trmat_table, tensor_table)

        @test SUNDMRG.save_block_and_trmat!(storage, :l, block, trmat) === nothing
        @test block_table[(:l, block.length)] === block
        @test trmat_table[(:l, block.length)] == trmat

        loaded_block, loaded_trmat = SUNDMRG.load_block_and_trmat(
            storage,
            :l,
            block.length,
            SUNDMRG.CPUEngine,
            Val(2),
        )
        @test loaded_block === block
        @test loaded_trmat == trmat
        @test all(x -> x isa Matrix{Float64}, loaded_trmat)
    end

    @testset "JLD2 storage" begin
        mktempdir() do scratch
            dirid = "finite_helpers"
            mkdir(joinpath(scratch, "temp$dirid"))
            storage = SUNDMRG.JLD2InternalStorage(scratch, dirid)

            @test SUNDMRG.save_block_and_trmat!(storage, :r, block, trmat) === nothing

            loaded_block, loaded_trmat = SUNDMRG.load_block_and_trmat(
                storage,
                :r,
                block.length,
                SUNDMRG.CPUEngine,
                Val(2),
            )
            @test loaded_block.length == block.length
            @test loaded_block.bonds == block.bonds
            @test repr(loaded_block.β_list) == repr(block.β_list)
            @test loaded_block.mβ_list_old == block.mβ_list_old
            @test loaded_block.mβ_list == block.mβ_list
            @test loaded_block.scalar_dict[:H] == block.scalar_dict[:H]
            @test loaded_block isa SUNDMRG.Block{2}
            @test loaded_trmat == trmat
            @test all(x -> x isa Matrix{Float64}, loaded_trmat)

            SUNDMRG.cleanup_storage!(storage)
            @test !isdir(joinpath(scratch, "temp$dirid"))
        end
    end
end

@testset "DMRG step result" begin
    step = SUNDMRG.DMRGStepResult(1, 2, 3, 4.0, 5.0, 6, 7, 8.0, 9, 10, nothing, nothing, nothing, nothing)
    @test step.block == 1
    @test step.es == 9
    @test step.Sj == 10
    @test step.env_block === nothing
    @test step.env_tensor_dict === nothing
    @test step.env_block_enl === nothing
    @test step.env_trmat === nothing

    env_step = SUNDMRG.DMRGStepResult(1, 2, 3, 4.0, 5.0, 6, 7, 8.0, 9, nothing, 10, 11, 12, 13)
    @test env_step.block == 1
    @test env_step.es == 9
    @test env_step.Sj === nothing
    @test env_step.env_block == 10
    @test env_step.env_tensor_dict == 11
    @test env_step.env_block_enl == 12
    @test env_step.env_trmat == 13
end
