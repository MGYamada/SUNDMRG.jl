using Pkg

# Julia 1.10 does not read [sources]. Resolve the sources in a disposable
# bootstrap environment, then carry their locations into the ED manifest.
repo = dirname(@__DIR__)
Pkg.activate(mktempdir())
Pkg.add(PackageSpec(url = "https://github.com/MGYamada/MAGMA.jl.git",
                    rev = "5545b1a27ee2516d9766c6a15238f006eceb1629"))
Pkg.develop(PackageSpec(path = repo))
Pkg.add(PackageSpec(name = "KrylovKit", version = "0.10.4"))
# A manifest carries source locations on both minimum and current Julia.
cp(joinpath(dirname(Base.active_project()), "Manifest.toml"),
   joinpath(@__DIR__, "Manifest.toml"); force = true)
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
