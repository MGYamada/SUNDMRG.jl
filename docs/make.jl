push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Documenter
using SUNDMRG

makedocs(;
    sitename = "SUNDMRG.jl",
    modules = [SUNDMRG, SUNDMRG.RepresentationTheory],
    checkdocs = :none,
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "DMRG Overview" => "dmrg_overview.md",
        "Usage" => "usage.md",
        "Examples" => "examples.md",
        "SU(Nc) Representation Theory" => "representation_theory.md",
        "Examples of SU(Nc) Representations" => "su_n_examples.md",
        "Representation Labels in SUNDMRG.jl" => "representation_notation.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/MGYamada/SUNDMRG.jl.git",
    devbranch = "main",
)
