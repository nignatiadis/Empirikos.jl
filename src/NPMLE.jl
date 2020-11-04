struct NPMLE{C} <: EBayesMethod
    convexclass::C
    solver::Any
    dict::Any
end

NPMLE(convexclass, solver; kwargs...) = NPMLE(convexclass, solver, kwargs)

struct FittedNPMLE{D, N<:NPMLE}
    prior::D
    npmle::N
    model::Any # add status?
end
Base.broadcastable(fitted_npmle::FittedNPMLE) = Ref(fitted_npmle)


marginalize(Z, fitted_npmle::FittedNPMLE) = marginalize(Z, fitted_npmle.prior)

function (target::EBayesTarget)(fitted_npmle::FittedNPMLE, args...)
    target(fitted_npmle.prior, args...)
end

function StatsBase.fit(npmle::NPMLE, Zs)
    Zs = summarize_by_default(Zs) ? summarize(Zs) : Zs

    convexclass = instantiate(npmle.convexclass, Zs; npmle.dict...)
    instantiated_npmle = @set npmle.convexclass = convexclass

    _fit(instantiated_npmle, Zs)
end

function _fit(npmle::NPMLE, Zs)
    @unpack convexclass, solver = npmle
    model = Model(solver)

    π = Empirikos.prior_variable!(model, convexclass)
    f = pdf.(π, Zs)

    _mult = multiplicity(Zs)
    n = length(_mult)

    @variable(model, u)

    @constraint(model,  vcat(u, f, _mult) in MathOptInterface.RelativeEntropyCone(2n+1))
    @objective(model, Min, u)
    optimize!(model)
    estimated_prior = convexclass(JuMP.value.(π.finite_param))
    FittedNPMLE(estimated_prior, npmle, model)
end

# NonparametricMLE( __ optional {ConvexPriorClass}; grid= , ngrid=  , method=:primal or :dual, solver= )

# NPMLE{}

# FModel()

#What are we estimating?
#How are we estimating it?
