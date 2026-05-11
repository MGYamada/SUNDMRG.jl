push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))

using Documenter
using SUNDMRG

makedocs(;
    sitename = "SUNDMRG.jl",
    modules = [SUNDMRG],
    checkdocs = :none,
    format = Documenter.HTML(; prettyurls = false),
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
