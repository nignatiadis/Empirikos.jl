abstract type ConvexMinimumDistanceMethod <: EBayesMethod end

struct NPMLE{C} <: ConvexMinimumDistanceMethod
    convexclass::C
    solver::Any
    dict::Any
end

NPMLE(convexclass, solver; kwargs...) = NPMLE(convexclass, solver, kwargs)

struct FittedConvexMinimumDistance{D,N<:ConvexMinimumDistanceMethod}
    prior::D
    method::N
    model::Any # add status?
end
Base.broadcastable(fitted_method::FittedConvexMinimumDistance) = Ref(fitted_method)
# TODO: replace these types of methods by a fit_if_not_fitted
StatsBase.fit(fitted_method::FittedConvexMinimumDistance, args...; kwargs...) =
    fitted_method


marginalize(Z, fitted_method::FittedConvexMinimumDistance) =
    marginalize(Z, fitted_method.prior)

function (target::EBayesTarget)(fitted_method::FittedConvexMinimumDistance, args...)
    target(fitted_method.prior, args...)
end

Distributions.pdf(fitted_method::FittedConvexMinimumDistance, Z) =
    Distributions.pdf(fitted_method.prior, Z)

# seems like template that could be useful more generally..
function StatsBase.fit(method::ConvexMinimumDistanceMethod, Zs; kwargs...)
    Zs = summarize_by_default(Zs) ? summarize(Zs) : Zs
    method = set_defaults(method, Zs; kwargs...)
    _fit(method, Zs)
end

function _fit(npmle::NPMLE, Zs)
    @unpack convexclass, solver = npmle
    model = Model(solver)

    π = Empirikos.prior_variable!(model, convexclass)
    f = pdf.(π, Zs)

    _mult = multiplicity(Zs)
    n = length(_mult)

    @variable(model, u)

    @constraint(model, vcat(u, f, _mult) in MathOptInterface.RelativeEntropyCone(2n + 1))
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
