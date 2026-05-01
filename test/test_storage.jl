@testset "Internal storage backends" begin
    block_table = Dict{Tuple{Symbol, Int}, Any}()
    trmat_table = Dict{Tuple{Symbol, Int}, Any}()
    tensor_table = Dict{Tuple{Symbol, Int, Int}, Any}()
    storage = SUNDMRG.MemoryInternalStorage(block_table, trmat_table, tensor_table)

    block = (length = 2, payload = "block")
    trmat = [[1.0 0.0; 0.0 1.0]]
    tensor = Matrix{Vector{Matrix{Float64}}}(undef, 1, 1)
    tensor[1, 1] = [[2.0 3.0]]

    SUNDMRG.save_block(storage, :l, 2, block)
    SUNDMRG.save_trmat(storage, :l, 2, trmat)
    SUNDMRG.save_tensor(storage, :l, 2, 1, tensor)

    @test SUNDMRG.load_block(storage, :l, 2) == block
    @test SUNDMRG.load_trmat(storage, :l, 2) == trmat
    @test SUNDMRG.load_tensor(storage, :l, 2, 1) == tensor
    @test SUNDMRG.load_tensor(storage, :l, 2, 1) === nothing

    @test SUNDMRG.init_internal_storage(false, ".", block_table, trmat_table, tensor_table, 0) isa SUNDMRG.MemoryInternalStorage

    mktempdir() do scratch
        dirid = "storage_test"
        mkdir(joinpath(scratch, "temp$dirid"))
        disk_storage = SUNDMRG.JLD2InternalStorage(scratch, dirid)

        SUNDMRG.save_block(disk_storage, :r, 3, block)
        SUNDMRG.save_trmat(disk_storage, :r, 3, trmat)
        SUNDMRG.save_tensor(disk_storage, :r, 3, 2, tensor)

        @test SUNDMRG.load_block(disk_storage, :r, 3) == block
        @test SUNDMRG.load_trmat(disk_storage, :r, 3) == trmat
        @test SUNDMRG.load_tensor(disk_storage, :r, 3, 2) == tensor
        @test SUNDMRG.load_tensor(disk_storage, :r, 3, 2) === nothing

        SUNDMRG.cleanup_storage!(disk_storage)
        @test !isdir(joinpath(scratch, "temp$dirid"))

        initialized = SUNDMRG.init_internal_storage(true, scratch, block_table, trmat_table, tensor_table, 0)
        @test initialized isa SUNDMRG.JLD2InternalStorage
        @test isdir(joinpath(scratch, "temp$(initialized.dirid)"))
        SUNDMRG.cleanup_storage!(initialized)
        @test !isdir(joinpath(scratch, "temp$(initialized.dirid)"))
    end
end
