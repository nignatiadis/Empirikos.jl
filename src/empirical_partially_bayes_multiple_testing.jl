abstract type LimmaMethod end

struct MomentLimma <: LimmaMethod end

"""
    WinsorizedLimma(; winsor_p = (0.05, 0.9))

Phipson-Smyth Winsorized method-of-moments estimator for limma's common
scaled inverse-χ² variance prior. `winsor_p` gives the lower and upper
clipping quantiles.
"""
Base.@kwdef struct WinsorizedLimma{T} <: LimmaMethod
    winsor_p::T = (0.05, 0.9)
end

"""
    QuantileLimma(; quantile_p = (0.2, 0.8), scale_p = 0.5)

Quantile-ratio estimator for limma's common scaled inverse-χ² variance prior.
The prior degrees of freedom are estimated by matching the ratio of two
sample-variance quantiles to the corresponding F-quantile ratio; the scale is
then estimated by matching the `scale_p` quantile.
"""
Base.@kwdef struct QuantileLimma{T,S} <: LimmaMethod
    quantile_p::T = (0.2, 0.8)
    scale_p::S = 0.5
end

"""
    Limma(; method = MomentLimma())

Estimator for the scaled inverse-χ² variance prior used by limma.
"""
struct Limma{M<:LimmaMethod}
    method::M
end

Limma(; method = MomentLimma()) = Limma(method)

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
    BinnedPrior(; base = DiscretePriorClass(), binning = identity)

Variance-prior wrapper that fits a separate prior for each value of
`binning.(Ms)`. The optional third argument to `fit(test, Zs, Ms)` is the
covariate. The default `binning = identity` treats `Ms` as a discrete covariate;
otherwise, `binning` may be a scalar function or a StatsDiscretizations
discretizer.
"""
Base.@kwdef struct BinnedPrior{B,F}
    base::B = DiscretePriorClass()
    binning::F = identity
end

struct FittedBinnedPrior{D<:StatsDiscretizations.Dictionary}
    store::D
end

Base.keys(prior::FittedBinnedPrior) = keys(prior.store)
Base.getindex(prior::FittedBinnedPrior, bin) = prior.store[bin]
(prior::FittedBinnedPrior)(bin) = prior[bin]

"""
    EmpiricalPartiallyBayesTTest(; multiple_test = BenjaminiHochberg(), α = 0.05, prior = DiscretePriorClass(), solver = Hypatia.Optimizer, discretize_marginal = false, prior_grid_size = 300, lower_quantile = 0.01)

Performs empirical partially Bayes multiple testing.

## Fields

- `multiple_test`: Multiple testing procedure from MultipleTesting.jl (default: `BenjaminiHochberg()`).
- `α`: Significance level (default: 0.05).
- `prior`: Prior distribution. Default: `DiscretePriorClass()`. Alternatives
  include `Empirikos.Limma()`,
  `Empirikos.Limma(method = Empirikos.WinsorizedLimma())`,
  `Empirikos.Limma(method = Empirikos.QuantileLimma())`,
  `Empirikos.TrendedPrior()`, `Empirikos.BinnedPrior()`, or a distribution
  from Distributions.jl. Note: Other fields are ignored if using these
  alternatives.
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

function _fit_ttest(test::EmpiricalPartiallyBayesTTest, prior::BinnedPrior, mu_hat, Ss, Ms)
    bins = _covariate_bins(prior, Ms, length(Ss))
    fitted_prior = _fit_binned_prior(test, prior.base, Ss, bins)
    pvalues = map(eachindex(mu_hat, Ss, bins)) do i
        limma_pvalue(mu_hat[i], Ss[i], fitted_prior[bins[i]])
    end

    (
        method = test,
        prior = fitted_prior,
        bin = bins,
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


fit_prior(::EmpiricalPartiallyBayesTTest, prior::Limma, Ss) =
    fit_prior(prior.method, Ss)

fit_prior(::MomentLimma, Ss) = Empirikos.fit_limma(Ss)

fit_prior(method::WinsorizedLimma, Ss) =
    Empirikos.fit_limma_robust(Ss; winsor_p = method.winsor_p)

fit_prior(method::QuantileLimma, Ss) =
    Empirikos.fit_limma_quantile(Ss; quantile_p = method.quantile_p, scale_p = method.scale_p)

function fit_prior(::EmpiricalPartiallyBayesTTest, prior::Distribution, Ss)
    prior
end

function _covariate_bins(prior::BinnedPrior, Ms, n)
    isnothing(Ms) &&
        throw(ArgumentError("BinnedPrior requires passing covariates to `fit(test, Zs, Ms)`."))
    length(Ms) == n ||
        throw(DimensionMismatch("`Ms` has length $(length(Ms)); expected $n."))

    bins = prior.binning === identity ? collect(Ms) : prior.binning.(Ms)
    any(ismissing, bins) &&
        throw(ArgumentError("BinnedPrior does not support missing bin labels."))

    bins
end

function _fit_binned_prior(test::EmpiricalPartiallyBayesTTest, prior, Ss, bins)
    levels = unique(bins)
    priors = map(levels) do level
        idx = findall(bin -> isequal(bin, level), bins)
        fit_prior(test, prior, Ss[idx])
    end

    FittedBinnedPrior(StatsDiscretizations.Dictionary(levels, priors))
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
