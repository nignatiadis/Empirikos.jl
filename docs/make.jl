using Documenter
using DocumenterCitations
using DocumenterTools: Themes
using Empirikos

Themes.compile(joinpath(@__DIR__,"src/assets/light.scss"), joinpath(@__DIR__,"src/assets/themes/documenter-light.css"))


bib = CitationBibliography(joinpath(@__DIR__, "ebayes.bib"))

makedocs(
    bib,
    sitename = "Empirikos",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [Empirikos],
    pages    = [
        "Introduction" => "index.md",
        "Manual" => ["samples.md",
            "convexpriors.md",
            "estimation.md",
            "flocalizations.md",
            "estimands.md",
            "intervals.md"]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nignatiadis/Empirikos.jl.git",
    versions = ["stable" => "v^", "v#.#", "dev" => "master"]
)
