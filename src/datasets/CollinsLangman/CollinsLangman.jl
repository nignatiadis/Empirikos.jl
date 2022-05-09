export CollinsLangman

"""
    CollinsLangman
"""
module CollinsLangman

using CSV
using ..Empirikos: NonCentralHypergeometricSample


const DATA = joinpath(@__DIR__, "collins_langman_1985.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    Zs = NonCentralHypergeometricSample.(tbl.XC, tbl.NC, tbl.NT, tbl.XC + tbl.XT)
end


end
