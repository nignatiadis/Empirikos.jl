export EfronMorrisBaseball

module EfronMorrisBaseball

using CSV
using ..Empirikos: NormalSample

const DATA = joinpath(@__DIR__, "EfronMorrisBB.txt")

function load_table()
    CSV.File(DATA)
end

function ebayes_samples(; variance_stabilized=true)
    bb = EfronMorrisBaseball.load_table()
    Zs1 = BinomialSample.(bb.Hits, bb.AtBats)
    Zs = StandardNormalSample.(sqrt(45).* asin.( (2. * response.(Zs1) ./ ntrials.(Zs1)) .-1))
    if variance_stabilized
        return Zs
    else
        return Zs1
    end
end

end
