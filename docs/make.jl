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
        "Getting Started" => [
            "Usage" => "usage.md",
            "Runtime Options" => "runtime_options.md",
            "Examples" => "examples.md",
        ],
        "Coefficient Tables" => "coefficient_tables.md",
        "Algorithm" => [
            "DMRG Overview" => "dmrg_overview.md",
            "SUNDMRG Algorithm" => "algorithm.md",
        ],
        "SU(Nc) Symmetry" => [
            "Representation Theory" => "representation_theory.md",
            "Representation Examples" => "su_n_examples.md",
            "Representation Labels" => "representation_notation.md",
        ],
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/MGYamada/SUNDMRG.jl.git",
    devbranch = "main",
)
