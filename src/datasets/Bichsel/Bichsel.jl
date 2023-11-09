export Bichsel

module Bichsel

using CSV
using ..Empirikos: PoissonSample, MultinomialSummary, Interval

const DATA = joinpath(@__DIR__, "bichsel.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples(; combine=false)
    tbl = load_table()
    zs = tbl.z
    Ns = tbl.count
    if combine
        Zs_keys = [PoissonSample.(0:4); PoissonSample(Interval(5,nothing))]
        Ns = [Ns[1:5]; Ns[6] + Ns[7]]
        Zs = MultinomialSummary(Zs_keys, Ns)
    else
        Zs_keys = PoissonSample.(zs)
    end
    Zs = MultinomialSummary(Zs_keys, Ns)
    Zs
end

end
