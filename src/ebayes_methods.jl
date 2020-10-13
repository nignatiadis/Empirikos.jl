abstract type EBayesMethod end

function StatsBase.fit(mom::EBayesMethod, Zs)
    StatsBase.fit(mom, Zs, skedasticity(Zs))
end

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


# struct SURE <: EBayesMethod
#   object::SoftThreshold,
#   location = :grandmean
# end

# SURE(Normal(); )       <---> keep dataset              SURE(Normal())          CovariateSample{EB, X}
# FixedLocationSURE( ; )      <---> keep datasaet
# GrandmeanLocationSURE( _ ; )   <----> keep dataset
# LinearModelLocationSURE( ; ); MLE(....)         @model :μ
