struct FoldedNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    FoldedNormal{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function FoldedNormal(μ::T, σ::T) where {T <: Real}
    return FoldedNormal{T}(μ, σ)
end


function Distributions.pdf(d::FoldedNormal, x::Real)
    d_normal = Normal(d.μ, d.σ)
    d_pdf = pdf(d_normal, x) + pdf(d_normal, -x)
    return x >= 0 ? d_pdf : zero(d_pdf)
end


function Distributions.logpdf(d::FoldedNormal, x::T) where {T<:Real}
    log(pdf(d, x))
end


function Distributions.cdf(d::FoldedNormal, x::Real)
    d_normal = Normal(d.μ, d.σ)
    d_cdf = cdf(d_normal, x) - cdf(d_normal, -x)
    return x > 0 ? d_cdf : zero(d_cdf)
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
    FoldedNormalSample(response(Z), std(Z))
end

response(Z::FoldedNormalSample) = Z.Z
nuisance_parameter(Z::FoldedNormalSample) = Z.σ
std(Z::FoldedNormalSample) = Z.σ
var(Z::FoldedNormalSample) = abs2(std(Z))


function _symmetrize(Zs::AbstractVector{<:FoldedNormalSample})
   random_signs =  2 .* rand(Bernoulli(), length(Zs)) .-1
   NormalSample.(random_signs .* response.(Zs), std.(Zs))
end
function likelihood_distribution(Z::FoldedNormalSample, μ)
    FoldedNormal(μ, nuisance_parameter(Z))
end

function Base.show(io::IO, Z::FoldedNormalSample{<:Real})
    Zz = response(Z)
    print(io, "Z=")
    print(io, lpad(round(Zz, digits = 4),8))
    print(io, " | ", "σ=")
    print(io, rpad(round(std(Z), digits = 3),5))
end

function Base.show(io::IO, Z::FoldedNormalSample{<:Interval})
    Zz = response(Z)
    print(io, "Z ∈ ")
    show(IOContext(io, :compact => true), Zz)
    print(io, " | ", "σ=")
    print(io, rpad(round(std(Z), digits = 3),5))
end

function default_target_computation(::BasicPosteriorTarget,
    ::FoldedNormalSample,
    ::Normal
)
    Conjugate()
end

function marginalize(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded = NormalSample(Z.Z, Z.σ)
    marginal_dbn = marginalize(Z_unfolded, prior)
    FoldedNormal(marginal_dbn.μ, marginal_dbn.σ)
end


function posterior(Z::FoldedNormalSample, prior::Normal)
    Z_unfolded = NormalSample(Z.Z, Z.σ)
    posterior_dbn = posterior(Z_unfolded, prior)
    FoldedNormal(posterior_dbn.μ, posterior_dbn.σ)
end
