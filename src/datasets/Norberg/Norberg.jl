export Norberg

module Norberg

using CSV
using ..Empirikos: PoissonSample

const DATA = joinpath(@__DIR__, "norberg.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    PoissonSample.(tbl.Death, tbl.Exposure ./ 344)
end

end
