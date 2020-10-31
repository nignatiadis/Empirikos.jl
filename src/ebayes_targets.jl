"""
	EBayesTarget

Abstract type that describes Empirical Bayes estimands (which we want to estimate or conduct inference for).
"""
abstract type EBayesTarget end

broadcastable(target::EBayesTarget) = Ref(target)


abstract type AbstractPosteriorTarget <: EBayesTarget end

abstract type AbstractTargetComputation end

struct Conjugate <: AbstractTargetComputation end

function default_target_computation end


function default_target_computation(sample, prior, target)
    default_target_computation(sample, prior)
end

function default_target_computation(sample, prior)
    default_target_computation(sample)
end



# ::Conjugate, ::LinearOverLinear, ::MixtureOfConjugates




abstract type LinearEBayesTarget <: EBayesTarget end

"""
	cf(::LinearEBayesTarget, t)

The characteristic function of ``L(\\cdot)``, a `LinearEBayesTarget`, which we define as follows:

For ``L(\\cdot)`` which may be written as ``L(G) = \\int \\psi(\\mu)dG\\mu``
(for a measurable function ``\\psi``) this returns the Fourier transform of ``\\psi``
evaluated at t, i.e., ``\\psi^*(t) = \\int \\exp(it x)\\psi(x)dx``.
Note that ``\\psi^*(t)`` is such that for distributions ``G`` with density ``g``
(and ``g^*`` the Fourier Transform of ``g``) the following holds:
```math
L(G) = \\frac{1}{2\\pi}\\int g^*(\\mu)\\psi^*(\\mu) d\\mu
```
"""
function cf(::LinearEBayesTarget, t) end

"""
	PriorDensity(z::Float64) <: LinearEBayesTarget
## Example call
```julia
PriorDensity(2.0)
```
## Description
This is the evaluation functional of the density of ``G`` at `z`, i.e.,
``L(G) = G'(z) = g(z)`` or in Julia code `L(G) = pdf(G, z)`.
"""
struct PriorDensity{T<:Real} <: LinearEBayesTarget
    μ::T
end

location(target::PriorDensity) = target.μ

function cf(target::PriorDensity, t)
    exp(im * location(target) * t)
end

function (target::PriorDensity)(μ::Number)
    location(target) == μ ? one(μ) : zero(μ)
end

function (target::PriorDensity)(prior::Distribution)
    pdf(prior, location(target))
end

Base.extrema(target::PriorDensity) = (0, Inf)


"""
	MarginalDensity(Z::EBayesSample) <: LinearEBayesTarget
## Example call
```julia
MarginalDensity(StandardNormalSample(2.0))
```
## Description
Describes the marginal density evaluated at ``Z=z``  (e.g. ``Z=2`` in the example above)
of a sample drawn from the hierarchical model
```math
\\mu \\sim G, Z \\sim \\mathcal{N}(0,1)
```
In other words, letting ``\\phi`` the Standard Normal pdf
```math
L(G) = \\phi \\star dG(z)
```
Note that `2.0` has to be wrapped inside `StandardNormalSample(2.0)` since this target
depends not only on `G` and the location, but also on the likelihood.
"""
struct MarginalDensity{T} <: LinearEBayesTarget
    Z::T
end

location(target::MarginalDensity) = target.Z

function (target::MarginalDensity)(μ::Number)
    likelihood(location(target), μ)
end

function (target::MarginalDensity)(prior::Distribution)
    pdf(prior, location(targetZ))
end



# Posterior Targets

function (target::AbstractPosteriorTarget)(prior::Distribution)
    _comp = default_target_computation(location(target), prior, target)
    target(prior, location(target), _comp)
end

function (targets::AbstractVector{<:EBayesTarget})(prior)
    [target(prior) for target in targets]
end

location(target::AbstractPosteriorTarget) = target.Z
Base.denominator(target::AbstractPosteriorTarget) = MarginalDensity(location(target))

struct PosteriorTargetNumerator{T} <: LinearEBayesTarget
    posterior_target::T
end

location(target::PosteriorTargetNumerator) = location(target.posterior_target)

function (post_numerator::PosteriorTargetNumerator)(prior)
    _post =   post_numerator.posterior_target
    post_numerator.posterior_target(prior)*denominator(post_)(prior)
end

Base.numerator(target::AbstractPosteriorTarget) = PosteriorTargetNumerator(target)



#function (postmean::PosteriorMean)(prior, Z::EBayesSample, ::Conjugate)
#    mean(posterior(Z, prior))
#end

"""
    PosteriorMean(Z::EBayesSample) <: EBayesTarget

Type representing the posterior mean, i.e.,

```math
E_G[\\mu_i \\mid Z_i = z]
```

"""
struct PosteriorMean{T} <: AbstractPosteriorTarget
    Z::T
end
PosteriorMean() = PosteriorMean(missing)

function (postmean::PosteriorMean)(prior, Z::EBayesSample, ::Conjugate)
    mean(posterior(Z, prior))
end






#
struct PosteriorVariance{T} <: AbstractPosteriorTarget
    Z::T
end
PosteriorVariance() = PosteriorVariance(missing)

function (postvar::PosteriorVariance)(prior, Z::EBayesSample, ::Conjugate)
    var(posterior(Z, prior))
end
