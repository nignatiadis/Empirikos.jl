"""
Abstract type representing empirical Bayes estimation methods.
"""
abstract type EBayesMethod end

function StatsBase.fit(mom::EBayesMethod, Zs)
    StatsBase.fit(mom, Zs, skedasticity(Zs))
end

struct MethodOfMoments{O} <: EBayesMethod
    object::O
    params::Any
end

MethodOfMoments(o) = MethodOfMoments(o, nothing)

Base.@kwdef struct ParametricMLE{O, S} <: EBayesMethod
    model::O
    solver::S
    kwargs::Any = nothing
end



# struct SURE <: EBayesMethod
#   object::SoftThreshold,
#   location = :grandmean
# end

# SURE(Normal(); )       <---> keep dataset              SURE(Normal())          CovariateSample{EB, X}
# FixedLocationSURE( ; )      <---> keep datasaet
# GrandmeanLocationSURE( _ ; )   <----> keep dataset
# LinearModelLocationSURE( ; ); MLE(....)         @model :μ
