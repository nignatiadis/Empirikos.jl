"""
    ScaledChiSquareSample(Z, ν)

An observed sample ``Z`` drawn from a scaled chi-square distribution with unknown scale ``\\sigma^2 > 0``.

```math
Z \\sim \\frac{\\sigma^2}{\\nu}}\\chi^2_{\\nu}
```

``\\sigma^2`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.
"""
struct ScaledChiSquareSample{T, S} <: ContinuousEBayesSample{T}
    Z::T
    ν::S
end

ScaledChiSquareSample(ν) = ScaledChiSquareSample(missing, ν)

eltype(Z::ScaledChiSquareSample{T}) where {T} = T
support(Z::ScaledChiSquareSample) = RealInterval(0, +Inf)

response(Z::ScaledChiSquareSample) = Z.Z
nuisance_parameter(Z::ScaledChiSquareSample) = Z.ν

function likelihood_distribution(Z::ScaledChiSquareSample, σ²)
    ν = Z.ν
    Gamma(ν/2,  σ² * 2 / ν)
end


function Base.show(io::IO, Z::ScaledChiSquareSample)
    resp_Z = response(Z)
    spaces_to_keep = 1
    spaces = repeat(" ", spaces_to_keep)
    print(io, "Z=", resp_Z, spaces, "| ", "ν=", Z.ν)
end

struct InverseScaledChiSquare{T, S} <: ContinuousUnivariateDistribution
    σ²::T
    ν ::S
end

function Distributions.InverseGamma(d::InverseScaledChiSquare)
    _shape = d.ν/2
    _scale = d.ν/2 * d.σ²
    InverseGamma(_shape, _scale)
end

function Distributions.pdf(d::InverseScaledChiSquare, x::Real)
   pdf(InverseGamma(d), x)
end


function Distributions.logpdf(d::InverseScaledChiSquare, x::T) where {T<:Real}
    logpdf(InverseGamma(d), x)
end


function Distributions.cdf(d::InverseScaledChiSquare, x::Real)
    cdf(InverseGamma(d), x)
end


# Conjugate computations

function default_target_computation(::BasicPosteriorTarget, ::ScaledChiSquareSample, ::InverseScaledChiSquare)
    Conjugate()
end

function marginalize(Z::ScaledChiSquareSample, prior::InverseScaledChiSquare)
    Distributions.AffineDistribution(0, prior.σ², FDist(Z.ν, prior.ν))
end

function posterior(Z::ScaledChiSquareSample, prior::InverseScaledChiSquare)
    aggregate_ν = Z.ν + prior.ν
    aggregate_σ² = (Z.Z * Z.ν  +  prior.σ² * prior.ν) / aggregate_ν
    InverseScaledChiSquare(aggregate_σ², aggregate_ν)
end


function limma_pvalue(β_hat, Z::ScaledChiSquareSample, prior::InverseScaledChiSquare)
    post = posterior(Z, prior)
    t_moderated = β_hat / sqrt.(post.σ²)
    2*ccdf(TDist(post.ν), abs(t_moderated))
end


function limma_pvalue(β_hat, Z::ScaledChiSquareSample, prior::DiscreteNonParametric)
    post = posterior(Z, prior)
    σs = sqrt.(support(post))
    πs = probs(post)

    pvals = 2 .* ccdf.(Normal.(0, σs), (abs(β_hat)))
    LinearAlgebra.dot(pvals, πs)
end
