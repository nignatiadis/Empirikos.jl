abstract type ConvexMinimumDistanceMethod <: EBayesMethod end

"""
    NPMLE(convexclass, solver) <: Empirikos.EBayesMethod

Given ``n`` independent samples ``Z_i`` from the empirical Bayes problem with prior ``G`` known to lie
in the `convexclass` ``\\mathcal{G}``, estimate ``G`` by Nonparametric Maximum Likelihood (NPMLE)
```math
\\widehat{G}_n \\in \\operatorname{argmax}_{G \\in \\mathcal{G}}\\left\\{\\sum_{i=1}^n \\log( f_{i,G}(Z_i)) \\right\\},
```
where ``f_{i,G}(z) = \\int p_i(z \\mid \\mu) dG(\\mu)`` is the marginal density of the ``i``-th sample.
The optimization is conducted by a JuMP compatible `solver`.
"""
struct NPMLE{C, S} <: ConvexMinimumDistanceMethod
    convexclass::C
    solver::S
    kwargs::Any
end

NPMLE(convexclass, solver; kwargs...) = NPMLE(convexclass, solver, kwargs)
NPMLE(;convexclass, solver, kwargs...) = NPMLE(convexclass, solver; kwargs)

function Base.show(io::IO, npmle::NPMLE)
    print(io, "NPMLE with ")
    show(io, npmle.solver)
    println(io, " and 𝒢:")
    show(io, npmle.convexclass)
end


struct FittedConvexMinimumDistance{D,N<:ConvexMinimumDistanceMethod}
    prior::D
    method::N
    model::Any # add status?
end
Base.broadcastable(fitted_method::FittedConvexMinimumDistance) = Ref(fitted_method)

function Base.show(io::IO, fittedmethod::FittedConvexMinimumDistance)
    print(io, "Fitted ")
    show(io, fittedmethod.method)
end

# TODO: replace these types of methods by a fit_if_not_fitted
StatsBase.fit(fitted_method::FittedConvexMinimumDistance, args...; kwargs...) =
    fitted_method


marginalize(Z, fitted_method::FittedConvexMinimumDistance) =
    marginalize(Z, fitted_method.prior)

function (target::EBayesTarget)(fitted_method::FittedConvexMinimumDistance, args...)
    target(fitted_method.prior, args...)
end


# TODO: macro to all of pdf, cdf,...
Distributions.pdf(fitted_method::FittedConvexMinimumDistance, Z) =
    Distributions.pdf(fitted_method.prior, Z)
Distributions.logpdf(fitted_method::FittedConvexMinimumDistance, Z) =
    Distributions.logpdf(fitted_method.prior, Z)


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
    # the output is equal to the marginal density (for each Z_i) up to a multiplicative constant
    # such an approach is more stable numerically
    f = rescaled_pdf.(π, Zs)

    _mult = multiplicity(Zs) ./ nobs(Zs)
    n = length(_mult)

    @variable(model, u)

    @constraint(model, vcat(u, f, _mult) in MathOptInterface.RelativeEntropyCone(2n + 1))
    @objective(model, Min, u)
    optimize!(model)
    check_moi_optimal(model)
    estimated_prior = π()
    FittedConvexMinimumDistance(estimated_prior, npmle, model)
end

"""
    KolmogorovSmirnovMinimumDistance(convexclass, solver) <: Empirikos.EBayesMethod

Given ``n`` i.i.d. samples from the empirical Bayes problem with prior ``G`` known to lie
in the `convexclass` ``\\mathcal{G}`` , estimate ``G`` as follows:
```math
\\widehat{G}_n \\in \\operatorname{argmin}_{G \\in \\mathcal{G}}\\{\\sup_{t \\in \\mathbb R}\\lvert F_G(t) - \\widehat{F}_n(t)\\rvert\\},
```
where ``\\widehat{F}_n`` is the ECDF of the samples. The optimization is conducted by a JuMP compatible `solver`.
"""
struct KolmogorovSmirnovMinimumDistance{C} <: ConvexMinimumDistanceMethod
    convexclass::C
    solver::Any
    dict::Any
end

function KolmogorovSmirnovMinimumDistance(convexclass, solver)
    KolmogorovSmirnovMinimumDistance(convexclass, solver, nothing)
end

function KolmogorovSmirnovMinimumDistance(; convexclass, solver)
    KolmogorovSmirnovMinimumDistance(convexclass, solver, nothing)
end

function Base.show(io::IO, ks::KolmogorovSmirnovMinimumDistance)
    print(io, "Kolmogorov-Smirnov Minimum Distance with ")
    show(io, ks.solver)
    println(io, " and 𝒢:")
    show(io, ks.convexclass)
end

function _fit(method::KolmogorovSmirnovMinimumDistance, Zs)
    @unpack convexclass, solver = method
    model = Model(solver)

    π = Empirikos.prior_variable!(model, convexclass)

    dkw = fit(DvoretzkyKieferWolfowitz(;max_constraints=Inf), Zs)

    F = cdf.(π, keys(dkw.summary))
    Fhat = collect(values(dkw.summary))

    @variable(model, u)

    @constraint(model, F - Fhat .<= u)
    @constraint(model, F - Fhat .>= -u)

    @objective(model, Min, u)
    optimize!(model)
    check_moi_optimal(model)
    estimated_prior = π()
    FittedConvexMinimumDistance(estimated_prior, method, model)
end
