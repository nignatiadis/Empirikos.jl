"""
Abstract type that describes Empirical Bayes estimands (which we want to estimate or conduct inference for).
"""
abstract type EBayesTarget end


function (targets::AbstractArray{<:EBayesTarget})(prior)
    [target(prior) for target in targets]
end

abstract type AbstractPosteriorTarget <: EBayesTarget end
abstract type BasicPosteriorTarget <: AbstractPosteriorTarget end

"""
    LinearEBayesTarget <: EBayesTarget

Abstract type that describes Empirical Bayes estimands that are linear functionals of the
prior `G`.
"""
abstract type LinearEBayesTarget <: EBayesTarget end

Base.extrema(::EBayesTarget) = (-Inf, +Inf) # allow distribution-dependent choice?
_support(::LinearEBayesTarget) = Interval(-Inf, +Inf)
location(::LinearEBayesTarget) = nothing


abstract type AbstractTargetComputation end

struct Conjugate <: AbstractTargetComputation end
struct NumeratorOfConjugate <: AbstractTargetComputation end
struct QuadgkQuadrature <: AbstractTargetComputation end

struct LinearOverLinear{N,D} <: AbstractTargetComputation
    num::N
    denom::D
end

Base.numerator(lin::LinearOverLinear) = lin.num
Base.denominator(lin::LinearOverLinear) = lin.denom



function (target::EBayesTarget)(prior)
    _loc = location(target) #TODO What about targets w/o location -> make it a trait!
    _comp = default_target_computation(target, _loc, prior)
    compute_target(_comp, target, _loc, prior)
end

function compute_target(comp::AbstractTargetComputation, target, prior)
    compute_target(comp, target, location(target), prior)
end



function default_target_computation(target::LinearEBayesTarget, sample, prior)
    QuadgkQuadrature()
end

function default_target_computation(target::AbstractPosteriorTarget, sample, prior)
    LinearOverLinear(nothing, nothing)
end

function compute_target(lin::LinearOverLinear, target::AbstractPosteriorTarget, sample, prior)
    # TODO: Actually take advantage of options in LinearOverLinear fields
    _num = numerator(target)(prior)
    _denom = denominator(target)(prior)
    _num/_denom
end

# TODO: Allow setting tolerances.
function compute_target(lin::QuadgkQuadrature, target::LinearEBayesTarget, sample, prior::ContinuousUnivariateDistribution)
   _interval = intersect(_support(target), _support(prior))
   _lower = leftendpoint(_interval)
   _upper = rightendpoint(_interval)
   quadgk( μ -> target(μ)*pdf(prior,μ), _lower, _upper)[1]
end



Base.@kwdef struct AffineTransformedLinearTarget{T, I, S} <: LinearEBayesTarget
    a::I = 0
    b::S = 1
    target::T
end

import Base.:*

function Base.:*(s::Number, t::LinearEBayesTarget)
    AffineTransformedLinearTarget(;b=s, target=t)
end

Base.:*(t::LinearEBayesTarget, s::Number) = s*t

(t::AffineTransformedLinearTarget)(prior::Union{<:Number, <:Distribution}) = t.b * t.target(prior) + t.a





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
function Distributions.cf(::LinearEBayesTarget, t) end

"""
	PriorDensity(z::Float64) <: LinearEBayesTarget
## Example call
```jldoctest
julia> PriorDensity(2.0)
PriorDensity{Float64}(2.0)
```
## Description
This is the evaluation functional of the density of ``G`` at `z`, i.e.,
``L(G) = G'(z) = g(z)`` or in Julia code `L(G) = pdf(G, z)`.
"""
struct PriorDensity{T} <: LinearEBayesTarget
    μ::T
end

location(target::PriorDensity) = target.μ

function Distributions.cf(target::PriorDensity{<:Real}, t)
    exp(im * location(target) * t)
end

#TODO: Not sure this is the right dispatch one(\mu) instead of e.g. one(Float64)
function (target::PriorDensity{<:Real})(μ::Number)
    location(target) == μ ? one(μ) : zero(μ)
end

function (target::PriorDensity{<:Interval})(μ::Number)
    in(μ, location(target)) ? one(μ) : zero(μ)
end

function (target::PriorDensity)(prior::Distribution)
    StatsDiscretizations.pdf(prior, location(target))
end

Base.extrema(target::PriorDensity) = (0, Inf)
Base.extrema(target::PriorDensity{<:AbstractInterval}) = (0, 1)


"""
	MarginalDensity(Z::EBayesSample) <: LinearEBayesTarget
## Example call
```julia
MarginalDensity(StandardNormalSample(2.0))
```
## Description
Describes the marginal density evaluated at ``Z=z``  (e.g. ``Z=2`` in the example above).
In the example above the sample is drawn from the hierarchical model
```math
\\mu \\sim G, Z \\sim \\mathcal{N}(0,1)
```
In other words, letting ``\\varphi`` the Standard Normal pdf
```math
L(G) = \\varphi \\star dG(z)
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
    pdf(prior, location(target))
end





location(target::AbstractPosteriorTarget) = target.Z


"""
    Base.denominator(target::AbstractPosteriorTarget)

Suppose a posterior target ``\\theta_G(z)``, such as the posterior mean can be written as:
```math
\\theta_G(z) = \\frac{ a_G(z)}{f_G(z)} = \\frac{ \\int h(\\mu)dG(\\mu)}{\\int p(z \\mid \\mu)dG(\\mu)}.
```

For example, for the posterior mean ``h(\\mu) =  \\mu \\cdot p(z \\mid \\mu)``. Then `Base.denominator`
returns the linear functional representing ``G \\mapsto f_G(z)`` (i.e., typically the marginal density).
Also see [`Base.numerator(::AbstractPosteriorTarget)`](@ref).
"""
Base.denominator(target::AbstractPosteriorTarget) = MarginalDensity(location(target))

struct PosteriorTargetNumerator{T} <: LinearEBayesTarget
    posterior_target::T
end

location(target::PosteriorTargetNumerator) = location(target.posterior_target)

function default_target_computation(target::PosteriorTargetNumerator{<:BasicPosteriorTarget}, sample, prior)
    _post_target = default_target_computation(target.posterior_target, sample, prior)
    _post_target == Conjugate() ? NumeratorOfConjugate() : QuadgkQuadrature()
end

function compute_target(::NumeratorOfConjugate, post_numerator::PosteriorTargetNumerator, sample, prior)
    _post = post_numerator.posterior_target
    post_numerator.posterior_target(prior) * denominator(_post)(prior)
end

function (post_numerator::PosteriorTargetNumerator)(μ::Number)
    _post = post_numerator.posterior_target
    post_numerator.posterior_target(μ) * denominator(_post)(μ)
end

"""
    Base.numerator(target::AbstractPosteriorTarget)

Suppose a posterior target ``\\theta_G(z)``, such as the posterior mean can be written as:
```math
\\theta_G(z) = \\frac{ a_G(z)}{f_G(z)} = \\frac{ \\int h(\\mu)dG(\\mu)}{\\int p(z \\mid \\mu)dG(\\mu)}.
```

For example, for the posterior mean ``h(\\mu) =  \\mu \\cdot p(z \\mid \\mu)``. Then `Base.numerator`
returns the linear functional representing ``G \\mapsto a_G(z)``.
"""
Base.numerator(target::AbstractPosteriorTarget) = PosteriorTargetNumerator(target)


struct PosteriorTargetNullHypothesis{T,S} <: LinearEBayesTarget
    posterior_target::T
    c::S
end

location(target::PosteriorTargetNullHypothesis) = location(target.posterior_target)

function (post_null::PosteriorTargetNullHypothesis)(prior::T) where {T<:Union{Distribution,Number}}
    c = post_null.c
    _post = post_null.posterior_target
    numerator(_post)(prior) - c * denominator(_post)(prior)
end



Base.@kwdef struct AffineTransformedPosteriorTarget{T, I, S} <: BasicPosteriorTarget
    a::I = 0
    b::S = 1
    target::T
end

import Base.:*

function Base.:*(s::Number, t::BasicPosteriorTarget)
    AffineTransformedPosteriorTarget(;b=s, target=t)
end

Base.:*(t::BasicPosteriorTarget, s::Number) = s*t

(t::AffineTransformedPosteriorTarget)(prior::Union{<:Number, <:Distribution}) = t.b * t.target(prior) + t.a

location(target::AffineTransformedPosteriorTarget) = location(target.target)

"""
    PosteriorDensity(Z::EBayesSample, μ) <: AbstractPosteriorTarget

Type representing the posterior density given Z at ``\\mu``, i.e.,

```math
p_G(\\mu \\mid Z_i = z)
```
"""
struct PosteriorDensity{T,S} <: BasicPosteriorTarget
    Z::T
    μ::S
end

function compute_target(::Conjugate, target::PosteriorDensity, Z::EBayesSample, prior)
    pdf(posterior(Z, prior), target.μ)
end

function (target::PosteriorDensity)(μ::Number)
    one(Float64)
end




"""
    PosteriorMean(Z::EBayesSample) <: AbstractPosteriorTarget

Type representing the posterior mean, i.e.,

```math
E_G[\\mu_i \\mid Z_i = z]
```
"""
struct PosteriorMean{T} <: BasicPosteriorTarget
    Z::T
end

function compute_target(::Conjugate, postmean::PosteriorMean, Z::EBayesSample, prior)
    mean(posterior(Z, prior))
end

function (postmean::PosteriorMean)(μ::Number)
    μ
end

function Base.show(io::IO, target::PosteriorMean)
    Z = target.Z
    param = primary_parameter(Z)
    print(io, "𝔼[", param," | ", Z,"]")
end

"""
    SymmetricPosteriorMean(Z::EBayesSample) <: AbstractPosteriorTarget

Type representing the symmetrirzed posterior mean, i.e.,

```math
\\EE[ \\Symm{G}]{\\mu \\mid Z=z}, where \\Symm{G} := \\varepsilon \\cdot \\mu,  \\varepsilon \\sim \\mathrm{Rademacher}, \\mu \\indep \\varepsilon
```
"""
struct SymmetricPosteriorMean{T} <: BasicPosteriorTarget
    Z::T
end

function compute_target(::Conjugate, target::SymmetricPosteriorMean, Z::EBayesSample, prior)
    reflected_prior = -1*prior
    symm_prior = MixtureModel([prior, reflected_prior], [0.5, 0.5])
    return compute_target(Conjugate(), PosteriorMean(Z), Z, symm_prior)
end
function (target::SymmetricPosteriorMean)(μ::Number)
    μ 
end

function Base.show(io::IO, target::SymmetricPosteriorMean)
    Z = target.Z
    param = primary_parameter(Z)
    print(io, "Symm𝔼[", param," | ", Z,"]")
end


"""
    PosteriorSecondMoment(Z::EBayesSample) <: AbstractPosteriorTarget

Type representing the second moment of the posterior centered around c, i.e.,

```math
E_G[(\\mu_i-c)^2 \\mid Z_i = z]
```
"""
struct PosteriorSecondMoment{T,S} <: BasicPosteriorTarget
    Z::T
    c::S
end

PosteriorSecondMoment(z) = PosteriorSecondMoment(z, zero(Float64))

function compute_target(::Conjugate, postmoment::PosteriorSecondMoment, Z::EBayesSample, prior)
    _post = posterior(Z, prior)
    c = postmoment.c
    var(_post) + abs2(mean(_post) - c)
end

function (postmean::PosteriorSecondMoment)(μ::Number)
    abs2(μ)
end


"""
    PosteriorVariance(Z::EBayesSample) <: AbstractPosteriorTarget

Type representing the posterior variance, i.e.,

```math
V_G[\\mu_i \\mid Z_i = z]
```

"""
struct PosteriorVariance{T} <: BasicPosteriorTarget
    Z::T
end
PosteriorVariance() = PosteriorVariance(missing)

function compute_target(::Conjugate, postmean::PosteriorVariance, Z::EBayesSample, prior)
    var(posterior(Z, prior))
end

Base.numerator(::PosteriorVariance) = throw("Posterior Variance is not fractional.")
Base.denominator(::PosteriorVariance) = throw("Posterior Variance is not fractional.")


"""
    PosteriorProbability(Z::EBayesSample, s) <: AbstractPosteriorTarget

Type representing the posterior probability, i.e.,

```math
\\Prob_G[\\mu_i \\in s \\mid Z_i = z]
```

"""
struct PosteriorProbability{T,S} <: BasicPosteriorTarget
    Z::T
    s::S
end


function compute_target(::Conjugate, postprob::PosteriorProbability, Z::EBayesSample, prior)
    StatsDiscretizations.pdf(posterior(Z, prior), postprob.s)
end

function (postprob::PosteriorProbability{T, <:Interval})(μ::Number) where {T}
    _interval = postprob.s
    eltype(_interval)(μ in postprob.s)
end


function Base.extrema(target::PosteriorProbability)
    (0.0, 1.0)
end

# TODO: once we define support for posterior targets as well?
function _support(target::PosteriorTargetNumerator{<:PosteriorProbability})
    target.posterior_target.s
end


# Some additional linear targets that we keep unexported for now.


struct PriorMean <: LinearEBayesTarget end

(target::PriorMean)(μ::Number) = μ
(target::PriorMean)(prior::Distribution) = mean(prior)

struct PriorSecondMoment <: LinearEBayesTarget end

(target::PriorSecondMoment)(μ::Number) = abs2(μ)
(target::PriorSecondMoment)(prior::Distribution) = abs2(mean(prior)) + var(prior)

# Add some targets that are useful for studying RCTs and predictive power.






# Plotting code

@recipe function f(targets::AbstractVector{<:EBayesTarget}, g)
    length(unique(typeof.(targets))) == 1 || error("Expected homogeneous targets")
    xs = Float64.(location.(targets))
    ys = targets.(g)

    background_color_legend --> :transparent
    foreground_color_legend --> :transparent

    seriestype --> :path
    seriescolor --> "#550133"

    xs, ys
end
