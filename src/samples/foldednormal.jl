struct FoldedNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    FoldedNormal{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function FoldedNormal(μ::T, σ::T) where {T <: Real}
    return FoldedNormal{T}(μ, σ)
end

FoldedNormal(d::Normal) = FoldedNormal(d.μ, d.σ)
Distributions.Normal(d::FoldedNormal) = Normal(d.μ, d.σ)

function Distributions.pdf(d::FoldedNormal, x::Real)
    d_normal = Normal(d.μ, d.σ)
    d_pdf = pdf(d_normal, x) + pdf(d_normal, -x)
    return x >= 0 ? d_pdf : zero(d_pdf)
end


function Distributions.logpdf(d::FoldedNormal, x::T) where {T<:Real}
    d_normal = Normal(d.μ, d.σ)
    d_logpdf_left = logpdf(d_normal, -x)
    d_logpdf_right = logpdf(d_normal, x)
    d_logpdf = LogExpFunctions.logaddexp(d_logpdf_left, d_logpdf_right)
    return x >= 0 ? d_logpdf :  oftype(d_logpdf, -Inf)
end


function Distributions.cdf(d::FoldedNormal, x::Real)
    d_normal = Normal(d.μ, d.σ)
    d_cdf = cdf(d_normal, x) - cdf(d_normal, -x)
    return x > 0 ? d_cdf : zero(d_cdf)
end

function Distributions.ccdf(d::FoldedNormal, x::Real)
    d_normal = Normal(d.μ, d.σ)
    d_cdf = ccdf(d_normal, x) + cdf(d_normal, -x)
    return x > 0 ? d_cdf : one(d_cdf)
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

FoldedNormalSample(Z) = FoldedNormalSample(Z, 1.0)
FoldedNormalSample() = FoldedNormalSample(missing)

function FoldedNormalSample(Z::AbstractNormalSample)
    FoldedNormalSample(abs(response(Z)), std(Z))
end

response(Z::FoldedNormalSample) = Z.Z
nuisance_parameter(Z::FoldedNormalSample) = Z.σ
std(Z::FoldedNormalSample) = Z.σ
var(Z::FoldedNormalSample) = abs2(std(Z))

function NormalSample(Z::FoldedNormalSample; positive_sign = true)
    response_Z = positive_sign ? response(Z) : -response(Z)
    NormalSample(response_Z, nuisance_parameter(Z))
end
function Base.show(io::IO, Z::FoldedNormalSample)
    print(io, "|", NormalSample(Z), "|")
end


function _symmetrize(Zs::AbstractVector{<:FoldedNormalSample})
   random_signs =  2 .* rand(Bernoulli(), length(Zs)) .-1
   NormalSample.(random_signs .* response.(Zs), std.(Zs))
end

function likelihood_distribution(Z::FoldedNormalSample, μ)
    FoldedNormal(μ, nuisance_parameter(Z))
end



function default_target_computation(::BasicPosteriorTarget,
    ::FoldedNormalSample,
    ::Normal
)
    Conjugate()
end

function marginalize(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded = NormalSample(Z)
    marginal_dbn = marginalize(Z_unfolded, prior)
    FoldedNormal(marginal_dbn.μ, marginal_dbn.σ)
end

function marginalize(Z::FoldedNormalSample, prior::FoldedNormal)
    marginalize(Z, Normal(prior))
end


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
