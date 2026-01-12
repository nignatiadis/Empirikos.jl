"""
    Folded{D<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution

A folded continuous univariate distribution, representing the absolute value of a random variable 
following the original distribution `D`. For a distribution `dist`, the folded version has:

- `pdf(d, x) = pdf(unfold(d), x) + pdf(unfold(d), -x)` for x ≥ 0  
- `cdf(d, x) =  P(unfold(d) ≤ x) − P(unfold(d) ≤ −x)` for x ≥ 0  
- analogously for `ccdf` and `quantile`

# Fields
- `dist::D`: The original (unfolded) distribution.
"""
struct Folded{D<:ContinuousUnivariateDistribution} <: ContinuousUnivariateDistribution
    dist::D
end

fold(dist::ContinuousUnivariateDistribution) = Folded(dist)
unfold(dist::Folded) = dist.dist

Base.minimum(::Folded) = zero(Float64)

function Distributions.maximum(d::Folded)
    orig_max = maximum(unfold(d))
    if isinf(orig_max)
        return Inf
    else
        return Float64(max(orig_max, -minimum(unfold(d))))
    end
end


function Distributions.pdf(d::Folded, x::Real)
    d_pdf = pdf(unfold(d), x) + pdf(unfold(d), -x)
    return x >= 0 ? d_pdf : zero(d_pdf)
end

function Distributions.logpdf(d::Folded, x::Real)
    d_logpdf_left = logpdf(unfold(d), -x)
    d_logpdf_right = logpdf(unfold(d), x)
    d_logpdf = LogExpFunctions.logaddexp(d_logpdf_left, d_logpdf_right)
    return x >= 0 ? d_logpdf : oftype(d_logpdf, -Inf)
end

function Distributions.cdf(d::Folded, x::Real)
    d_cdf = cdf(unfold(d), x) - cdf(unfold(d), -x)
    return x > 0 ? d_cdf : zero(d_cdf)
end

function Distributions.logcdf(d::Folded, x::Real)
    if x ≤ 0
        return oftype(float(x), -Inf)  
    elseif x == Inf
        return oftype(float(x), 0.0)    
    end

    lx  = logcdf(unfold(d),  x)
    lmx = logcdf(unfold(d), -x)
    LogExpFunctions.logsubexp(lx, lmx)
end

function Distributions.ccdf(d::Folded, x::Real)
    d_ccdf = ccdf(unfold(d), x) + cdf(unfold(d), -x)
    return x > 0 ? d_ccdf : one(d_ccdf)
end

"""
    quantile(d::Folded, q::Real)

Compute the quantile of a folded normal by using the noncentral chi-squared distribution.
"""
function Distributions.quantile(d::Folded{<:Normal}, q::Real)
    orig_normal = d.dist
    μ = mean(orig_normal)  
    σ = std(orig_normal)  
    σ == 0 && return abs(μ)
    
    λ = (μ/σ)^2
    nc_chisq = NoncentralChisq(1, λ)
    
    σ * sqrt(quantile(nc_chisq, q))
end

"""
    quantile(d::Folded{<:TDist}, q::Real)

Compute the quantile for folded t-distributions, this maps to the 
`(1 + q)/2` quantile of the original symmetric distribution.
"""
function Distributions.quantile(d::Folded{<:TDist}, q::Real)
    Distributions.quantile(unfold(d), (1+q)/2)
end

"""
    quantile(d::Folded{<:LocationScale{T, Continuous, <:TDist}}, q::Real) where T

Compute the quantile for a folded location-scale t-distribution.
Currently only supports the case where μ = 0 (symmetric about origin).
"""
function Distributions.quantile(d::Folded{<:LocationScale{T, Continuous, <:TDist}}, q::Real) where T
    ls = d.dist
    μ, σ = ls.μ, ls.σ
    tdist = ls.ρ
    
    μ == 0 || throw(ArgumentError("quantile for Folded LocationScale TDist only supported when μ = 0, got μ = $μ"))
    
    σ * quantile(tdist, (1 + q) / 2)
end



"""
    FoldedNormalSample(Z,σ)

An observed sample ``Z`` equal to the absolute value of a draw
from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z = |Y|, Y\\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.
"""
struct FoldedNormalSample{T,S} <: ContinuousEBayesSample{T}
    Z::T
    σ::S
end

function FoldedNormalSample(Z::Real, σ::Real)
    z = float(Z); s = float(σ)
    z ≥ 0      || throw(DomainError(Z, "Folded observation requires Z ≥ 0."))
    (isfinite(s) && s > 0) || throw(DomainError(σ, "σ must be finite and > 0."))
    return FoldedNormalSample{typeof(z), typeof(s)}(z, s)
end

FoldedNormalSample(Z) = FoldedNormalSample(Z, 1.0)
FoldedNormalSample() = FoldedNormalSample(missing)

function FoldedNormalSample(Z::AbstractNormalSample)
    FoldedNormalSample(abs(response(Z)), std(Z))
end

response(Z::FoldedNormalSample) = Z.Z
nuisance_parameter(Z::FoldedNormalSample) = Z.σ
std(Z::FoldedNormalSample) = Z.σ
var(Z::FoldedNormalSample) = abs2(std(Z))

"""
    NormalSample(Z::FoldedNormalSample; positive_sign = true)

Convert a `FoldedNormalSample` back to a `NormalSample` by assigning a sign (default: positive).
"""
function NormalSample(Z::FoldedNormalSample; positive_sign = true)
    response_Z = positive_sign ? response(Z) : -response(Z)
    NormalSample(response_Z, nuisance_parameter(Z))
end
function Base.show(io::IO, Z::FoldedNormalSample)
    print(io, "|", NormalSample(Z), "|")
end

"""
    _symmetrize(Zs::AbstractVector{<:FoldedNormalSample})

Randomly assign signs to a vector of FoldedNormalSample to reconstruct a vector of `NormalSample`s. 
"""
function _symmetrize(Zs::AbstractVector{<:FoldedNormalSample})
   random_signs =  2 .* rand(Bernoulli(), length(Zs)) .-1
   NormalSample.(random_signs .* response.(Zs), std.(Zs))
end

"""
    likelihood_distribution(Z::FoldedNormalSample, μ)

Return the folded Normal distribution `fold(Normal(μ, σ))` for a given mean `μ`.
"""
function likelihood_distribution(Z::FoldedNormalSample, μ)
    fold(Normal(μ, nuisance_parameter(Z)))
end



function default_target_computation(::BasicPosteriorTarget,
    ::FoldedNormalSample,
    ::Normal
)
    Conjugate()
end

# perhaps this can apply to more general folded samples.

"""
    marginalize(Z::FoldedNormalSample, prior::Normal) -> Folded{Normal}

Compute marginal distribution for folded normal observation with normal prior.

1. Unfold the observation to normal sample
2. Compute marginal distribution of the normal sample with the normal prior
3. Fold the resulting distribution
"""
function marginalize(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded = NormalSample(Z)
    fold(marginalize(Z_unfolded, prior)) 
end

function marginalize(Z::FoldedNormalSample, prior::Folded{<:Normal})
    marginalize(Z, unfold(prior))
end

"""
    posterior(Z::FoldedNormalSample, prior::Normal) -> MixtureModel{Normal}

Compute the posterior distribution for a folded normal observation by considering both possible 
signs of the original observation, weighted by their marginal probabilities under the prior.
"""
function posterior(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded_positive = NormalSample(Z; positive_sign = true)
    Z_unfolded_negative = NormalSample(Z; positive_sign = false)

    marginal_prob_positive = pdf(prior, Z_unfolded_positive)
    marginal_prob_negative = pdf(prior, Z_unfolded_negative)
    marginal_prob_sum = marginal_prob_positive + marginal_prob_negative

    prob_positive = marginal_prob_positive / marginal_prob_sum
    prob_negative = marginal_prob_negative / marginal_prob_sum

    posterior_dbn_positive = posterior(Z_unfolded_positive, prior)
    posterior_dbn_negative = posterior(Z_unfolded_negative, prior)

    posterior_dbn = MixtureModel(
        [posterior_dbn_positive, posterior_dbn_negative],
        [prob_positive, prob_negative]
    )
    posterior_dbn
end






"""
    marginalize(Z::FoldedNormalSample, prior::Uniform) -> Folded{UniformNormal}

Compute the marginal distribution for a folded normal observation under a uniform prior.

# Arguments
- `Z::FoldedNormalSample`: A folded normal observation.
- `prior::Uniform`: Uniform prior distribution.

# Returns
- `Folded{UniformNormal}`: Folded UniformNormal distribution representing the marginal distribution.
"""
function marginalize(Z::FoldedNormalSample, prior::Uniform)
    Z_unfolded = NormalSample(Z)
    unif_normal = marginalize(Z_unfolded, prior)
    fold(unif_normal)
end

function (t::SignAgreementProbabilityNumerator{<:FoldedNormalSample})(prior::Distribution)
    z_val = t.Z.Z
    Z_plus = NormalSample(t.Z)
    Z_minus = NormalSample(t.Z; positive_sign=false)
    
    positive_set = Interval{:open,:open}(0.0, Inf)
    negative_set = Interval{:open,:open}(-Inf, 0.0)
    prob_positive = numerator(PosteriorProbability(Z_plus, positive_set))(prior)
    prob_negative = numerator(PosteriorProbability(Z_minus, negative_set))(prior)
    
    prob_positive + prob_negative
end


function (t::ReplicationProbability_num{<:FoldedNormalSample})(prior::Distribution)
    z_abs = t.Z.Z 
    lower = Distributions.minimum(prior)
    upper = Distributions.maximum(prior)
    term_plus, _ = quadgk(μ -> begin
        pdf(Normal(μ, 1), z_abs) * ccdf(Normal(μ, 1), 1.96) * pdf(prior, μ)
    end, lower, upper)
    
    term_minus, _ = quadgk(μ -> begin
        pdf(Normal(μ, 1), -z_abs) * cdf(Normal(μ, 1), -1.96) * pdf(prior, μ)
    end, lower, upper)
    
    return term_plus + term_minus
end

function (t::FutureCoverageProbability_num{<:FoldedNormalSample})(prior::Distribution)
    z_abs = t.Z.Z
    lower = Distributions.minimum(prior)
    upper = Distributions.maximum(prior)
    term_plus, _ = quadgk(μ -> begin
        pdf(Normal(μ, 1), z_abs) * (cdf(Normal(μ, 1), z_abs+1.96) - cdf(Normal(μ, 1), z_abs-1.96)) * pdf(prior, μ)
    end, lower, upper)
    
    term_minus, _ = quadgk(μ -> begin
        pdf(Normal(μ, 1), -z_abs) * (cdf(Normal(μ, 1), -z_abs+1.96) - cdf(Normal(μ, 1), -z_abs-1.96)) * pdf(prior, μ)
    end, lower, upper)
    
    return term_plus + term_minus
end
