using Documenter
using Quarto

Quarto.render(joinpath(@__DIR__, "src"))

Documenter.deploydocs(repo = "git@github.com:nignatiadis/Empirikos.jl")
