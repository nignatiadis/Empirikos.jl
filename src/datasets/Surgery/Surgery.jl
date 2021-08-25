export Surgery

module Surgery

using CSV
using ..Empirikos: BinomialSample

const DATA = joinpath(@__DIR__, "surgery.csv")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples()
    tbl = load_table()
    Zs = BinomialSample.(tbl.s, tbl.n)
end

end
