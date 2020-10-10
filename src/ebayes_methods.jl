abstract type EBayesMethod end

struct MethodOfMoments{O} <:EBayesMethod
    object::O
    params
end

MethodOfMoments(o) = MethodOfMoments(o, nothing)

struct ParametricMLE{O} <:EBayesMethod
    object::O
    params
end

ParametricMLE(o) = ParametricMLE(o, nothing)



function StatsBase.fit(mom::EBayesMethod, Zs)
    StatsBase.fit(mom, Zs, skedasticity(Zs))
end
