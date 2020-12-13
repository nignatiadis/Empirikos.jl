using Documenter
using DocumenterCitations
using Empirikos

bib = CitationBibliography(joinpath(@__DIR__, "ebayes.bib"))

makedocs(
    bib,
    sitename = "Empirikos",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [Empirikos],
    pages    = [
        "index.md",
        "samples.md"
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nignatiadis/Empirikos.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)
