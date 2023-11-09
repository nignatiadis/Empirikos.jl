export ArceoGomezCamposVasquez

module ArceoGomezCamposVasquez

using CSV
using ..Empirikos: BinomialSample, BivariateBinomialSample, MultinomialSummary

const DATA = joinpath(@__DIR__, "agcv_counts.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples(; combine=false)
    tbl = load_table()
    Z1s = BinomialSample.(tbl.C_w, tbl.N_w)
    Z2s = BinomialSample.(tbl.C_b, tbl.N_b)
    Zs = BivariateBinomialSample.(Z1s, Z2s)
    MultinomialSummary(Zs, tbl.F)
end


end
