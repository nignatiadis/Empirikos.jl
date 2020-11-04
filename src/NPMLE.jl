abstract type ConvexMinimumDistanceMethod <: EBayesMethod end

struct NPMLE{C} <: ConvexMinimumDistanceMethod
    convexclass::C
    solver::Any
    dict::Any
end

NPMLE(convexclass, solver; kwargs...) = NPMLE(convexclass, solver, kwargs)

struct FittedConvexMinimumDistance{D, N<:ConvexMinimumDistanceMethod}
    prior::D
    method::N
    model::Any # add status?
end
Base.broadcastable(fitted_method::FittedConvexMinimumDistance) = Ref(fitted_method)


marginalize(Z, fitted_method::FittedConvexMinimumDistance) = marginalize(Z, fitted_method.prior)

function (target::EBayesTarget)(fitted_method::FittedConvexMinimumDistance, args...)
    target(fitted_method.prior, args...)
end

function StatsBase.fit(method::ConvexMinimumDistanceMethod, Zs)
    Zs = summarize_by_default(Zs) ? summarize(Zs) : Zs

    convexclass = instantiate(method.convexclass, Zs; method.dict...)
    instantiated_method = @set method.convexclass = convexclass

    _fit(instantiated_method, Zs)
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
    FittedConvexMinimumDistance(estimated_prior, npmle, model)
end


struct KolmogorovSmirnovMinimumDistance{C} <: ConvexMinimumDistanceMethod
    convexclass::C
    solver::Any
    dict::Any
end

function _fit(method::KolmogorovSmirnovMinimumDistance, Zs)
    @unpack convexclass, solver = method
    model = Model(solver)

    π = Empirikos.prior_variable!(model, convexclass)

    dkw = fit(DvoretzkyKieferWolfowitz(), Zs)

    F = cdf.(π, keys(dkw.summary))
    Fhat = collect(values(dkw.summary))

    @variable(model, u)

    @constraint(model, F - Fhat .<= u)
    @constraint(model, F - Fhat .>= -u)

    @objective(model, Min, u)
    optimize!(model)
    estimated_prior = convexclass(JuMP.value.(π.finite_param))
    FittedConvexMinimumDistance(estimated_prior, method, model)
end
# NonparametricMLE( __ optional {ConvexPriorClass}; grid= , ngrid=  , method=:primal or :dual, solver= )

# NPMLE{}

# FModel()

#What are we estimating?
#How are we estimating it?
