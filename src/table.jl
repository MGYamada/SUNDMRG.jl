"""
make_table(Nc, widthmax)
Making tables dictionary. MPI is not necessary.
"""
function make_table(Nc, widthmax)
    @load "table4half_SU$(Nc)_$widthmax.jld2" table4
    @load "table3nuhalf_SU$(Nc)_$widthmax.jld2" table_3ν
    tables = table_9ν(Nc, widthmax, table4, table_3ν)
    @save "table_SU$(Nc)_$widthmax.jld2" tables
end