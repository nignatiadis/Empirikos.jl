abstract type EBayesMethod end

struct MethodOfMoments{O} <:EBayesMethod
    object::O
    params
end

MethodOfMoments(o) = MethodOfMoments(o, nothing)

function StatsBase.fit(mom::MethodOfMoments, Zs)
    StatsBase.fit(mom, Zs, skedasticity(Zs))
end
