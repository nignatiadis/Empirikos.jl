abstract type AbstractNormalSample{T} <: EBayesSample{T} end

"""
    NormalSample(Z,σ)

A observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z \\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```julia
NormalSample(0.5, 1.0)          #Z=0.5, σ=1
```
"""
struct NormalSample{T,S} <: AbstractNormalSample{T}
	Z::T
 	σ::S
end

function NormalSample(σ::S) where {S}
	NormalSample(missing, σ)
end



"""
    StandardNormalSample(Z)

A observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 =1``.

```math
Z \\sim \\mathcal{N}(\\mu, 1)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```julia
StandardNormalSample(0.5)          #Z=0.5
```
"""
struct StandardNormalSample{T} <: AbstractNormalSample{T}
    Z::T
end


eltype(Z::AbstractNormalSample{T}) where {T} = T
support(Z::AbstractNormalSample) = RealInterval(-Inf, +Inf)

response(Z::AbstractNormalSample) = Z.Z
var(Z::AbstractNormalSample) = Z.σ^2
var(Z::StandardNormalSample) = one(eltype(response(Z)))

std(Z::AbstractNormalSample) = Z.σ
std(Z::StandardNormalSample) = one(eltype(response(Z)))

nuisance_parameter(Z::AbstractNormalSample) = std(Z)

likelihood_distribution(Z::AbstractNormalSample, μ) = Normal(μ, std(Z))
likelihood_distribution(Z::AbstractNormalSample) = likelihood_distribution(μ, zero(std(Z)))

function Base.isless(a::AbstractNormalSample, b::AbstractNormalSample)
    std(a) <= std(b) && response(a) < response(b)
end



# Targets

# TODO: Note this is not correct for intervals.
function cf(target::MarginalDensity{<:AbstractNormalSample}, t)
    error_dbn = likelihood_distribution(location(target))
    cf(error_dbn, t)
end


# Conjugate computations
function default_target_computation(::AbstractPosteriorTarget, ::AbstractNormalSample, ::Normal)
    Conjugate()
end

function marginalize(Z::AbstractNormalSample, prior::Normal)
    prior_var = var(prior)
    prior_μ = mean(prior)
    likelihood_var = var(Z)
    marginal_σ = sqrt(likelihood_var + prior_var)
    Normal(prior_μ, marginal_σ)
end


function posterior(Z::AbstractNormalSample, prior::Normal)
    z = response(Z)
    sigma_squared = var(Z)
    prior_mu = mean(prior)
    prior_A = var(prior)

    post_mean = (prior_A)/(prior_A + sigma_squared)*z + sigma_squared/(prior_A + sigma_squared)*prior_mu
    post_var = prior_A * sigma_squared / (prior_A + sigma_squared)
    Normal(post_mean, sqrt(post_var))
end
