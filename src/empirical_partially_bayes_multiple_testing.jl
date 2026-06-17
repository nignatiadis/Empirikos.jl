struct Limma end

"""
    NaturalSplineVarianceTrend(; df = nothing, min_variance = 1e-12, intercept = true)

Specification for fitting a natural spline to `log(S²)` as a function of `Ms`.
The fitting method is provided by the Splines2 package extension.
"""
Base.@kwdef struct NaturalSplineVarianceTrend{D,T}
    df::D = nothing
    min_variance::T = 1.0e-12
    intercept::Bool = true
end

"""
    TrendedPrior(; base = DiscretePriorClass(), trend = NaturalSplineVarianceTrend())

Variance-prior wrapper that first removes a log-variance trend. `base` may be a
prior specification and `trend` may be a trend specification or a callable fitted
trend. The fit result exposes the fitted base prior as `prior` and the callable
fitted trend as `trend`.
"""
Base.@kwdef struct TrendedPrior{B,T}
    base::B = DiscretePriorClass()
    trend::T = NaturalSplineVarianceTrend()
end

"""
    EmpiricalPartiallyBayesTTest(; multiple_test = BenjaminiHochberg(), α = 0.05, prior = DiscretePriorClass(), solver = Hypatia.Optimizer, discretize_marginal = false, prior_grid_size = 300, lower_quantile = 0.01)

Performs empirical partially Bayes multiple testing.

## Fields

- `multiple_test`: Multiple testing procedure from MultipleTesting.jl (default: `BenjaminiHochberg()`).
- `α`: Significance level (default: 0.05).
- `prior`: Prior distribution. Default: `DiscretePriorClass()`. Alternatives include `Empirikos.Limma()`, `Empirikos.TrendedPrior()`, or a distribution from Distributions.jl. Note: Other fields are ignored if using these alternatives.
- `solver`: Optimization solver (default: `Hypatia.Optimizer`). Not used with alternative `prior` choices.
- `discretize_marginal`: If true, discretizes marginal distribution (default: false). Not used with alternative `prior` choices.
- `prior_grid_size`: Grid size for prior distribution (default: 300). Not used with alternative `prior` choices.
- `lower_quantile`: Lower quantile for sample variances (default: 0.01).


## References

@ignatiadis2023empirical
"""
Base.@kwdef struct EmpiricalPartiallyBayesTTest
    multiple_test = BenjaminiHochberg()
    α = 0.05
    prior = DiscretePriorClass()
    solver
    discretize_marginal = false
    prior_grid_size = 300
    lower_quantile = 0.01
end

function Base.show(io::IO, test::EmpiricalPartiallyBayesTTest)
    α_str = "α=$(test.α)"
    prior_str = "prior=$(typeof(test.prior).name.name)"
    test_str = "mtp=$(typeof(test.multiple_test).name.name)"
    print(io, "EmpiricalPartiallyBayesTTest($α_str, $prior_str, $test_str)")
end

"""
    fit(test::EmpiricalPartiallyBayesMultipleTest, Zs::AbstractArray{<:NormalChiSquareSample})

Fit the empirical partially Bayes multiple testing model.

## Arguments

  - `test`: An `EmpiricalPartiallyBayesMultipleTest` object.
  - `Zs`: An array of `NormalChiSquareSample` objects.

## Returns

A named tuple containing the following fields:

  - `method`: The `EmpiricalPartiallyBayesMultipleTest` object.
  - `prior`: The estimated prior distribution.
  - `pvalue`: An array of empirical partially Bayes p-values.
  - `cutoff`: The cutoff value (such that all hypotheses with pvalue ≤ cutoff are rejected).
  - `adjp`: An array of adjusted p-values.
  - `rj_idx`: An array of rejection indicators.
  - `total_rejections`: The total number of rejections.

"""
function fit(
    test::EmpiricalPartiallyBayesTTest,
    Zs::AbstractArray{<:NormalChiSquareSample},
    Ms = nothing,
)
    mu_hat = getproperty.(Zs, :Z)
    Ss = ScaledChiSquareSample.(Zs)

    _fit_ttest(test, test.prior, mu_hat, Ss, Ms)
end

function _fit_ttest(test::EmpiricalPartiallyBayesTTest, prior, mu_hat, Ss, Ms)
    fitted_prior = fit_prior(test, prior, Ss)
    pvalues = limma_pvalue(mu_hat, Ss, fitted_prior)

    (
        method = test,
        prior = fitted_prior,
        _multiple_testing_result(test, pvalues)...,
    )
end

function _fit_ttest(test::EmpiricalPartiallyBayesTTest, prior::TrendedPrior, mu_hat, Ss, Ms)
    fitted_trend = fit_trend(prior.trend, Ms, response.(Ss))
    fitted_logvariance = fitted_trend.(Ms)
    Ss_trended = _rescale_variances(Ss, fitted_logvariance)
    fitted_prior = fit_prior(test, prior.base, Ss_trended)
    mu_hat_trended = mu_hat ./ exp.(fitted_logvariance ./ 2)
    pvalues = limma_pvalue(mu_hat_trended, Ss_trended, fitted_prior)

    (
        method = test,
        prior = fitted_prior,
        trend = fitted_trend,
        fitted_logvariance = fitted_logvariance,
        _multiple_testing_result(test, pvalues)...,
    )
end

function _multiple_testing_result(test, pvalues)
    adjp = adjust(pvalues, test.multiple_test)
    rj_idx = adjp .<= test.α
    total_rejections = sum(rj_idx)
    cutoff = iszero(total_rejections) ? zero(Float64) : maximum(pvalues[rj_idx])

    (
        pvalue = pvalues,
        cutoff = cutoff,
        adjp = adjp,
        rj_idx = rj_idx,
        total_rejections = total_rejections,
    )
end

limma_pvalue(mu_hat::AbstractArray, Ss::AbstractArray{<:ScaledChiSquareSample}, prior) =
    limma_pvalue.(mu_hat, Ss, Ref(prior))

function fit_prior(test::EmpiricalPartiallyBayesTTest, prior::DiscretePriorClass, Ss)

    prior = autoconvexclass(
        prior,
        Ss;
        prior_grid_size = test.prior_grid_size,
        lower_quantile = test.lower_quantile,
    )

    _npmle = NPMLE(prior, test.solver)

    if test.discretize_marginal
        disc = RealLineDiscretizer{:open,:closed}(support(prior))
        Ss_summary = summarize(disc.(Ss))
        npmle_prior = fit(_npmle, Ss_summary)
    else
        npmle_prior = fit(_npmle, Ss)
    end
    clean(npmle_prior.prior)
end


function fit_prior(::EmpiricalPartiallyBayesTTest, prior::Limma, Ss)
    Empirikos.fit_limma(Ss)
end

function fit_prior(::EmpiricalPartiallyBayesTTest, prior::Distribution, Ss)
    prior
end

function _rescale_variances(Ss, logscale)
    length(logscale) == length(Ss) ||
        throw(DimensionMismatch("fitted log-variance trend has length $(length(logscale)); expected $(length(Ss))."))

    ScaledChiSquareSample.(response.(Ss) ./ exp.(logscale), dof.(Ss))
end

function fit_trend(trend, Ms, s²s)
    if isnothing(Ms)
        throw(ArgumentError("A callable fitted trend requires passing `Ms` to `fit(test, Zs, Ms)`."))
    elseif isempty(Ms)
        throw(ArgumentError("`Ms` must be nonempty."))
    elseif applicable(trend, first(Ms))
        trend
    else
        throw(ArgumentError(
            "trend must be a supported unfitted trend specification or a callable fitted trend. " *
            "For NaturalSplineVarianceTrend, run `using Splines2` before fitting.",
        ))
    end
end

# Baseline: Regular t-test

Base.@kwdef struct SimultaneousTTest
    multiple_test =  BenjaminiHochberg()
    α::Float64 = 0.05
end

function fit(test::SimultaneousTTest, samples)
    mu_hat = getproperty.(samples, :Z)
    Ss = ScaledChiSquareSample.(samples)

    pvalues = 2*ccdf.(TDist.(dof.(Ss)), abs.(mu_hat) ./ sqrt.(response.(Ss)))
    _multiple_testing_result(test, pvalues)
end
