struct Limma end

"""
    EmpiricalPartiallyBayesTTest(; multiple_test = BenjaminiHochberg(), α = 0.05, prior = DiscretePriorClass(), solver = Hypatia.Optimizer, discretize_marginal = false, prior_grid_size = 300, lower_quantile = 0.01)

Performs empirical partially Bayes multiple testing.

## Fields

- `multiple_test`: Multiple testing procedure from MultipleTesting.jl (default: `BenjaminiHochberg()`).
- `α`: Significance level (default: 0.05).
- `prior`: Prior distribution. Default: `DiscretePriorClass()`. Alternatives include `Empirikos.Limma()` or a distribution from Distributions.jl. Note: Other fields are ignored if using these alternatives.
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
function fit(test::EmpiricalPartiallyBayesTTest, Zs::AbstractArray{<:NormalChiSquareSample})
    mu_hat = getproperty.(Zs, :Z)
    Ss = ScaledChiSquareSample.(Zs)

    prior = fit_prior(test, test.prior, Ss)

    multiple_test = test.multiple_test
    α = test.α

    pvalues = limma_pvalue.(mu_hat, Ss, prior)
    adjp = adjust(pvalues, multiple_test)
    rj_idx = adjp .<= α
    total_rejections = sum(rj_idx)

    if iszero(total_rejections)
        cutoff = zero(Float64)
    else
        cutoff = maximum(pvalues[rj_idx])
    end

    (
        method = test,
        prior = prior,
        pvalue = pvalues,
        cutoff = cutoff,
        adjp = adjp,
        rj_idx = rj_idx,
        total_rejections = total_rejections,
    )
end


function fit_prior(test::EmpiricalPartiallyBayesTTest, prior::DiscretePriorClass, Ss)

    prior = autoconvexclass(
        test.prior,
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

# Baseline: Regular t-test

Base.@kwdef struct SimultaneousTTest
    multiple_test =  BenjaminiHochberg()
    α::Float64 = 0.05
end

function fit(test::SimultaneousTTest, samples)
    mu_hat = getproperty.(samples, :Z)
    Ss = ScaledChiSquareSample.(samples)

    multiple_test = test.multiple_test
    α = test.α
    pvalues = 2*ccdf.(TDist.(dof.(Ss)), abs.(mu_hat) ./ sqrt.(response.(Ss)))
    adjp = adjust(pvalues, multiple_test)
    rj_idx = adjp .<= α
    total_rejections = sum(rj_idx)
    if iszero(total_rejections)
        cutoff = zero(Float64)
    else
        cutoff = maximum(pvalues[rj_idx])
    end

    (
    pvalue = pvalues,
    cutoff = cutoff,
    adjp = adjp,
    rj_idx = rj_idx,
    total_rejections = total_rejections
    )
end