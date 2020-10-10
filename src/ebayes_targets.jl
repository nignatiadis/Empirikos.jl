abstract type EBayesTarget end

abstract type AbstractPriorTarget <: EBayesTarget end
abstract type AbstractPosteriorTarget <: EBayesTarget end

abstract type AbstractTargetComputation end

struct Conjugate <: AbstractTargetComputation end

function default_target_computation end


function (target::AbstractPosteriorTarget)(prior::Distribution)
    _comp = default_target_computation(target, location(target), prior)
    target(prior, _comp)
end

function (targets::AbstractVector{<:EBayesTarget})(prior)
	[target(prior) for target in targets]
end

location(target::AbstractPosteriorTarget) = target.Z

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

struct PosteriorVariance{T} <: AbstractPosteriorTarget
    Z::T
end
PosteriorVariance() = PosteriorVariance(missing)

function (postvar::PosteriorVariance)(prior, Z::EBayesSample, ::Conjugate)
    var(posterior(Z, prior))
end


# ::Conjugate, ::LinearOverLinear, ::MixtureOfConjugates
