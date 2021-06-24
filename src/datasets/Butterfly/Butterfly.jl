export Butterfly

module Butterfly

using CSV
using ..Empirikos: MultinomialSummary, TruncatedPoissonSample

const DATA = joinpath(@__DIR__, "butterfly.txt")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    MultinomialSummary(TruncatedPoissonSample.(tbl.x), tbl.y)
end

end
