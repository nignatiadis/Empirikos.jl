export BertrandMullainathan

module BertrandMullainathan

using CSV
using ..Empirikos: BinomialSample, BivariateBinomialSample, summarize

const DATA = joinpath(@__DIR__, "bm_counts.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    Z1s = BinomialSample.(tbl.C_w, tbl.N_w)
    Z2s = BinomialSample.(tbl.C_b, tbl.N_b)
    Zs = BivariateBinomialSample.(Z1s, Z2s)
    summarize(Zs, tbl.F)
end


end
