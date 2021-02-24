export Neighborhoods

"""
    Neighborhoods

## The impact of Neighborhoods: Moving to opportunity

The reference for this dataset is the following:
>   Raj Chetty and Nathaniel Hendren.
The impacts of neighborhoods on intergenerational mobility II: County-level estimates.
The Quarterly Journal of Economics, 133(3):1163– 1228, 2018.
"""
module Neighborhoods

using CSV
using ..Empirikos: NormalSample

const DATA = joinpath(@__DIR__, "neighborhoods.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    Zs = MultinomialSummary(BinomialSample.(tbl.x, 20), tbl.N1)
end

end
