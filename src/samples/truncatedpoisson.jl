# actually define the distribution.

struct ZeroTruncatedPoisson{T<:Real} <: DiscreteUnivariateDistribution
    λ::T
    ZeroTruncatedPoisson{T}(λ::Real) where {T <: Real} = new{T}(λ)
end


function ZeroTruncatedPoisson(λ::T; check_args=true) where {T <: Real}
    check_args && Distributions.@check_args(Poisson, λ > zero(λ))
    return ZeroTruncatedPoisson{T}(λ)
end

function Distributions.pdf(d::ZeroTruncatedPoisson, x::Real)
    pois = Poisson(d.λ)
    pois_pdf = Distributions.pdf(pois, x)
    pois_zero = Distributions.ccdf(pois, 0)
    return x > 0 ? pois_pdf/pois_zero : zero(pois_pdf)
end

function _logpdf(d::ZeroTruncatedPoisson, x::T) where {T<:Real}
    pois = Poisson(d.λ)
    pois_zero = logccdf(pois, 0)
    if x >= 1
        logpdf(pois, x) - pois_zero
    else
        TF = float(T)
        -TF(Inf)
    end
end

function Distributions.logpdf(d::ZeroTruncatedPoisson, x::T) where {T<:Real}
    isinteger(x) || return -T(Inf)
    return _logpdf(d, x)
end


function _cdf(d::ZeroTruncatedPoisson, x::Real)
    pois = Poisson(d.λ)
    pois_pdf = Distributions.pdf(pois, zero(x))
    pois_cdf = Distributions.cdf(pois, x) - pois_pdf
    pois_zero = 1-pois_pdf
    return x > 0 ? pois_cdf/pois_zero : zero(pois_cdf)
end

Distributions.cdf(d::ZeroTruncatedPoisson, x::Real) = _cdf(d,x)
Distributions.cdf(d::ZeroTruncatedPoisson, x::Integer) = _cdf(d,x)




"""
    TruncatedPoissonSample(Z, E)

An observed sample ``Z`` drawn from a truncated Poisson distribution,

```math
Z \\sim \\text{Poisson}(\\mu \\cdot E)  \\mid Z \\geq 1.
```

The multiplying intensity ``E`` is assumed to be known (and equal to `1.0` by default), while
``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```julia
TruncatedPoissonSample(3)
TruncatedPoissonSample(3, 1.5)
```
"""
struct TruncatedPoissonSample{T,S} <: DiscreteEBayesSample{T}
    Z::T
    E::S
end

TruncatedPoissonSample(Z) = TruncatedPoissonSample(Z, 1.0)
TruncatedPoissonSample() = TruncatedPoissonSample(missing)

response(Z::TruncatedPoissonSample) = Z.Z
nuisance_parameter(Z::TruncatedPoissonSample) = Z.E
primary_parameter(::TruncatedPoissonSample) = :μ

function likelihood_distribution(Z::TruncatedPoissonSample, λ)
    ZeroTruncatedPoisson(λ * nuisance_parameter(Z))
end

summarize_by_default(Zs::Vector{<:TruncatedPoissonSample}) = skedasticity(Zs) == Homoskedastic()

function Base.show(io::IO, Z::TruncatedPoissonSample)
    resp_Z = response(Z)
    E = nuisance_parameter(Z)
    μ_string = E==1 ? "μ" : "μ⋅$(E)"
    print(io, "𝒫ℴ𝒾[>0](", resp_Z,"; ",  μ_string,")")
end


struct MarginallyTruncatedPoissonSample{T,S} <: DiscreteEBayesSample{T}
    Z::T
    E::S
end

MarginallyTruncatedPoissonSample(Z) = MarginallyTruncatedPoissonSample(Z, 1.0)
summarize_by_default(Zs::Vector{<:MarginallyTruncatedPoissonSample}) = skedasticity(Zs) == Homoskedastic()

response(Z::MarginallyTruncatedPoissonSample) = Z.Z
nuisance_parameter(Z::MarginallyTruncatedPoissonSample) = Z.E

function pdf(prior::Distribution, Z::MarginallyTruncatedPoissonSample)
    pois_Z = PoissonSample(Z.Z, Z.E)
    pois_0 = PoissonSample(0.0, Z.E)
    pdf(prior, pois_Z)/(1-pdf(prior, pois_0))
end

function loglikelihood(Z::MarginallyTruncatedPoissonSample, prior::Distribution)
    pois_Z = PoissonSample(Z.Z, Z.E)
    pois_0 = PoissonSample(0.0, Z.E)
    log_Z = loglikelihood(pois_Z, prior)
    log_0 = loglikelihood(pois_0, prior)
    prob_0 = exp(log_0)
    log_Z - log(1-prob_0)
end

function Base.show(io::IO, Z::MarginallyTruncatedPoissonSample)
    resp_Z = response(Z)
    E = nuisance_parameter(Z)
    μ_string = E==1 ? "μ" : "μ⋅$(E)"
    print(io, "𝒫ℴ𝒾(", resp_Z,"; ",  μ_string,")[>0]")
end
