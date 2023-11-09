export Shakespeare

module Shakespeare

using CSV
using ..Empirikos: MultinomialSummary, TruncatedPoissonSample

const DATA = joinpath(@__DIR__, "shakespeare.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    MultinomialSummary(TruncatedPoissonSample.(tbl.Zs), tbl.ns)
end

end
