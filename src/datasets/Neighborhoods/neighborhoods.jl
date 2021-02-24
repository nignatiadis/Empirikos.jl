export Neighborhoods

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
