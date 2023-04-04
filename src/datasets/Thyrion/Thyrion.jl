export Thyrion

"""
    Thyrion

    LA ROYALE BELGE

    A statistic covering vehicles in the category 'Tourism and Business' and
    belonging to the 2 lower classes of the rate, observed all during an entire year,
    gave the following results in which:


"""
module Thyrion

using CSV
using ..Empirikos: PoissonSample, MultinomialSummary

const DATA = joinpath(@__DIR__, "thyrion.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    zs = tbl.z
    Ns = tbl.count
    Zs_keys = PoissonSample.(zs)
    MultinomialSummary(Zs_keys, Ns)
end

end
