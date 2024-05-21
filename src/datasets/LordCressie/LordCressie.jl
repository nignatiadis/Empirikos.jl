export LordCressie

module LordCressie

using CSV
using ..Empirikos: summarize, BinomialSample

const DATA = joinpath(@__DIR__, "lord_cressie_1975.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    summarize(BinomialSample.(tbl.x, 20), tbl.N1)
end

end
