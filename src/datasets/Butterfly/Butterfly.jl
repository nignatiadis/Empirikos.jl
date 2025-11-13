export Butterfly

module Butterfly

using CSV
using ..Empirikos: summarize, TruncatedPoissonSample

const DATA = joinpath(@__DIR__, "butterfly.txt")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    summarize(TruncatedPoissonSample.(tbl.x), tbl.y)
end

end
