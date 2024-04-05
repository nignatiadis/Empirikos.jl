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

FoldedNormal(d::Normal) = FoldedNormal(d.μ, d.σ)
Distributions.Normal(d::FoldedNormal) = Normal(d.μ, d.σ)


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

function Distributions.ccdf(d::Folded, x::Real)
    d_ccdf = ccdf(unfold(d), x) + cdf(unfold(d), -x)
    return x > 0 ? d_ccdf : one(d_ccdf)
end


function Distributions.quantile(d::Folded, q::Real)
    Distributions.quantile(unfold(d), (1+q)/2)
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
    fold(Normal(μ, nuisance_parameter(Z)))
end



function default_target_computation(::BasicPosteriorTarget,
    ::FoldedNormalSample,
    ::Normal
)
    Conjugate()
end

# perhaps this can apply to more general folded samples.
function marginalize(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded = NormalSample(Z)
    fold(marginalize(Z_unfolded, prior)) 
end

function marginalize(Z::FoldedNormalSample, prior::Folded{Normal})
    marginalize(Z, unfold(prior))
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








function marginalize(Z::FoldedNormalSample, prior::Uniform)
    Z_unfolded = NormalSample(Z)
    -prior.a != prior.b && throw(DomainError(prior, "Code currently requires symmetric uniform distribution"))
    unif_normal = marginalize(Z_unfolded, prior)
    fold(unif_normal)
end