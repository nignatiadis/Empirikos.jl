export ArceoGomezCamposVasquez

module ArceoGomezCamposVasquez

using CSV
using ..Empirikos: BinomialSample, BivariateBinomialSample, summarize

const DATA = joinpath(@__DIR__, "agcv_counts.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    Z1s = BinomialSample.(tbl.C_f, tbl.N_f) # Female
    Z2s = BinomialSample.(tbl.C_m, tbl.N_m) # Male
    Zs = BivariateBinomialSample.(Z1s, Z2s)
    summarize(Zs, tbl.F)
end


end
