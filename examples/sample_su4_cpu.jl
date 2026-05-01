using SUNDMRG
using JLD2

function main()
    Nc = 4
    widthmax = 9
    @load "../jld2/table_SU$(Nc)_$widthmax.jld2" tables
    Lx = parse(Int, ARGS[1])
    Ly = parse(Int, ARGS[2])
    correlation = Lx < 30 ? :nn : :chain
    rank, dmrg = run_DMRG(SU(Nc)HeisenbergModel(), HoneycombLattice(Lx, Ly, :ZC), 100, [100, 200, 400, 800], 1600, CPUEngine; widthmax = widthmax, tables = tables, fileio = true, correlation = correlation, margin = 5, alg = :fast)
    if rank == 0
        @save "su$(Nc)_$(Lx)_$Ly.jld2" dmrg
        println("finished!")
    end
end
main()
