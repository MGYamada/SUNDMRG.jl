using SUNDMRG
using JLD2

function main()
    Nc = 4
    widthmax = 9
    @load "../jld2/table_SU$(Nc)_$widthmax.jld2" tables
    Lx = parse(Int, ARGS[1])
    Ly = parse(Int, ARGS[2])
    correlation = Lx < 30 ? :nn : :chain
    rank, dmrg = run_DMRG(SU(Nc)HeisenbergModel(), HoneycombLattice(Lx, Ly, :ZC), (100, 1e-5), [(100, 1e-5), (100, 1e-5), (200, 1e-6), (200, 1e-6), (400, 1e-7), (400, 1e-7), (800, 1e-8), (800, 1e-8), (1600, 1e-9), (1600, 1e-9), (3200, 1e-10), (3200, 1e-10), (6400, 1e-11), (6400, 1e-11)], (12800, 0.0), GPUEngine; widthmax = widthmax, tables = tables, fileio = true, correlation = correlation, margin = 5, alg = :fast)
    if rank == 0
        @save "su$(Nc)_$(Lx)_$Ly.jld2" dmrg
        println("finished!")
    end
end
main()