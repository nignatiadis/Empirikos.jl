"""
    BinomialSample(Z, n)

An observed sample ``Z`` drawn from a Binomial distribution with `n` trials.

```math
Z \\sim \\text{Binomial}(n, p)
```

``p`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``p``.

```jldoctest
julia> BinomialSample(2, 10)          # 2 out of 10 trials successful
ℬ𝒾𝓃(2; p, n=10)
```
"""
struct BinomialSample{T,S<:Integer} <: DiscreteEBayesSample{T}
    Z::T
    n::S     #TODO: add checks that Z \in {0,...,n}
end

BinomialSample(n::Integer) = BinomialSample(missing, n)

function Base.show(io::IO, Z::BinomialSample)
    resp_Z = response(Z)
    n = nuisance_parameter(Z)
    print(io, "ℬ𝒾𝓃(", resp_Z,"; ", primary_parameter(Z),", n=", n,")")
end

primary_parameter(::BinomialSample) = :p

summarize_by_default(::AbstractVector{<:BinomialSample}) = true



response(Z::BinomialSample) = Z.Z
ntrials(Z::BinomialSample) = Z.n
nuisance_parameter(Z::BinomialSample) = ntrials(Z)

likelihood_distribution(Z::BinomialSample, p) = Binomial(ntrials(Z), p)



function dictfun(::Nothing, Zs_summary::VectorOrSummary{<:DiscreteEBayesSample}, f)
    skedasticity(Zs_summary) == Homoskedastic() ||
        error("Heteroskedastic likelihood not implemented.")
    Zs = fill_levels(Zs_summary)
    if !isa(f, AbstractVector)
        f = f.(Zs)
    end
    DictFunction(Zs, f)
end


#-----------------------------------------------------------------------
#---------- Beta Binomial Conjugacy-------------------------------------
#-----------------------------------------------------------------------

function default_target_computation(::BasicPosteriorTarget,::BinomialSample, ::Beta)
    Conjugate()
end

function marginalize(Z::BinomialSample, prior::Beta)
    @unpack α, β = prior
    BetaBinomial(ntrials(Z), α, β)
end

function posterior(Z::BinomialSample, prior::Beta)
    Beta(prior.α + response(Z), prior.β + ntrials(Z) - response(Z))
end

# Fit BetaBinomial
function StatsBase.fit(
    ::MethodOfMoments{<:Beta},
    Zs::VectorOrSummary{<:BinomialSample},
    ::Homoskedastic,
)
    # TODO: Let ::Homoskedastic carry type information.
    n = ntrials(Zs[1])
    μ₁ = mean(response.(Zs), weights(Zs))
    μ₂ = mean(abs2.(response.(Zs)), weights(Zs))
    denom = n * (μ₂ / μ₁ - μ₁ - 1) + μ₁
    α = (n * μ₁ - μ₂) / denom
    β = (n - μ₁) * (n - μ₂ / μ₁) / denom
    Beta(α, β)
end







"""
    ChiSquaredFLocalization(α) <: FLocalization

The ``\\chi^2`` F-localization at confidence level ``1-\\alpha`` for a discrete random variable
taking values in ``\\{0,\\dotsc, N\\}``. It is equal to:
```math
f: \\sum_{x=0}^N \\frac{(n \\hat{f}_n(x) - n f(x))^2}{n f(x)} \\leq \\chi^2_{N,1-\\alpha},
```
where ``\\chi^2_{N,1-\\alpha}`` is the ``1-\\alpha`` quantile of the Chi-squared
distribution with ``N`` degrees of freedom, ``n`` is the sample size,
``\\hat{f}_n(x)`` is the proportion of samples equal to ``x`` and ``f(x)`` is then
population pmf.
"""
Base.@kwdef struct ChiSquaredFLocalization{T,S} <: FLocalization
    α::T = 0.05
    discretizer::S = nothing
end

vexity(::ChiSquaredFLocalization) = ConvexVexity()


struct FittedChiSquaredFLocalization{T,S,D<:StatsDiscretizations.Dictionary{T,S},C} <: FittedFLocalization
    summary::D
    band::S
    chisq::C
    dof::Int
    n::Int
end

vexity(chisq::FittedChiSquaredFLocalization) = vexity(chisq.chisq)

function nominal_alpha(chisq::FittedChiSquaredFLocalization)
    nominal_alpha(chisq.chisq)
end

# TODO: Allow this to work more broadly.
function StatsBase.fit(chisq::ChiSquaredFLocalization, Zs::AbstractVector{<:BinomialSample})
    StatsBase.fit(chisq, summarize(Zs))
end

function StatsBase.fit(chisq::ChiSquaredFLocalization, Zs_summary::MultinomialSummary)
    skedasticity(Zs_summary) == Homoskedastic() ||
    error("Heteroskedastic likelihood not implemented.")

    n = nobs(Zs_summary)
    first_sample = Zs_summary[1]
    if isnothing(chisq.discretizer)
        chisq = @set chisq.discretizer = FiniteGridDiscretizer(0:ntrials(first_sample))
    end
    _discr = set_response.(first_sample, chisq.discretizer)
    _dof = length(_discr) - 1  #again maybe Homoskedastic() should return what type of homoskedastic
    empirical_probs = Zs_summary.(_discr) ./ n
    _dict = StatsDiscretizations.Dictionary(_discr, empirical_probs)
    α = nominal_alpha(chisq)
    band =  quantile(Chisq(_dof), 1-α)
    FittedChiSquaredFLocalization(_dict, band, chisq, _dof, n)
end



function flocalization_constraint!(
    model,
    chisq::FittedChiSquaredFLocalization,
    prior::PriorVariable,
)
    n = chisq.n

    ts = @variable(model, [1:(chisq.dof+1)])
    @constraint(model, ts .>= 0)
    band = chisq.band
    for (i, (Z, pdf_value)) in enumerate(zip(keys(chisq.summary), values(chisq.summary)))
        _pdf = pdf(prior, Z::EBayesSample)
        @constraint(model, [ts[i]; n * _pdf; n * _pdf - n * pdf_value] in RotatedSecondOrderCone())
    end
    @constraint(model, sum(ts) <= band/2)
    model
end
