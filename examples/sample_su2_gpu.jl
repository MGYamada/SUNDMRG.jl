using SUNDMRG
using JLD2

function main()
    Nc = 2
    Lx = parse(Int, ARGS[1])
    Ly = parse(Int, ARGS[2])
    rank, dmrg = run_DMRG(SU(Nc)HeisenbergModel(), SquareLattice(Lx, Ly), 100, [100, 200, 400, 800], 1600, GPUEngine)
    if rank == 0
        @save "su$(Nc)_$(Lx)_$Ly.jld2" dmrg
        println("finished!")
    end
end
main()
