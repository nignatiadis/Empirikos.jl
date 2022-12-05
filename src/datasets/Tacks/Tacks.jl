export Tacks

module Tacks

using CSV
using ..Empirikos: BinomialSample

const DATA = joinpath(@__DIR__, "tacks.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    Zs = BinomialSample.(tbl.x, tbl.k)
end

end
