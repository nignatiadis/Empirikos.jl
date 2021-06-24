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
    pois_zero = 1-Distributions.pdf(pois, zero(x))
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
    isinteger(x) || return -TF(Inf)
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

function likelihood_distribution(Z::TruncatedPoissonSample, λ)
    ZeroTruncatedPoisson(λ * nuisance_parameter(Z))
end

summarize_by_default(Zs::Vector{<:TruncatedPoissonSample}) = skedasticity(Zs) == Homoskedastic()


function Base.show(io::IO, Z::TruncatedPoissonSample)
    resp_Z = response(Z)
    if ismissing(resp_Z) || isa(resp_Z, Interval)
        spaces_to_keep = 1
    else
        spaces_to_keep = max(3 - ndigits(response(Z)), 1)
    end
    spaces = repeat(" ", spaces_to_keep)
    print(io, "Z=", resp_Z, spaces, "| ", "E=", Z.E)
end
