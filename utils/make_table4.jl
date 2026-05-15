using SUNDMRG

function main()
    Nc, widthmax = parse_args()
    make_table4(Nc, widthmax)
end

function parse_args(args = ARGS)
    if length(args) == 0
        return 4, 9
    elseif length(args) == 2
        return parse(Int, args[1]), parse(Int, args[2])
    end
    throw(ArgumentError("usage: julia --project=. utils/make_table4.jl [Nc widthmax]"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
