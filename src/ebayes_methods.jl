abstract type EBayesMethod end

function StatsBase.fit(mom::EBayesMethod, Zs)
    StatsBase.fit(mom, Zs, skedasticity(Zs))
end

struct MethodOfMoments{O} <: EBayesMethod
    object::O
    params::Any
end

MethodOfMoments(o) = MethodOfMoments(o, nothing)

struct ParametricMLE{O} <: EBayesMethod
    object::O
    params::Any
end

ParametricMLE(o) = ParametricMLE(o, nothing)


# Some defaults for parametric distribution fitting
_default_constraints(::Normal) = TwiceDifferentiableConstraints([-Inf; 0.0], [Inf; Inf])
_default_init(::Normal) = [0.0; 1.0]

function _default_constraints(::Union{Beta,Gamma})
    TwiceDifferentiableConstraints([0.0; 0.0], [Inf; Inf])
end

function _default_init(::Union{Beta,Gamma})
    [1.0; 1.0]
end

# struct SURE <: EBayesMethod
#   object::SoftThreshold,
#   location = :grandmean
# end

# SURE(Normal(); )       <---> keep dataset              SURE(Normal())          CovariateSample{EB, X}
# FixedLocationSURE( ; )      <---> keep datasaet
# GrandmeanLocationSURE( _ ; )   <----> keep dataset
# LinearModelLocationSURE( ; ); MLE(....)         @model :μ
