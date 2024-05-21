struct TruncatedSample{T, EB<:ContinuousEBayesSample{T}, S} <: ContinuousEBayesSample{T}
    Z::EB
    truncation_set::S
end

nuisance_parameter(Z::TruncatedSample) = Z.truncation_set
primary_parameter(Z::TruncatedSample) = primary_parameter(Z.Z)

response(Z::TruncatedSample) = response(Z.Z)

function Base.show(io::IO, Z::TruncatedSample)
    print(io, "Trunc{", Z.Z," to ", string(Z.truncation_set),"}")
end

function _truncated(d::Distribution, interval::AbstractInterval)
    (isleftclosed(interval) && isinf(rightendpoint(d))) || throw("Only [a, ∞) {left-closed, right-unbounded} intervals allowed currently.")
    Distributions.truncated(d, first(interval), nothing)
end

function likelihood_distribution(Z::TruncatedSample{<:Any,<:Any,<:AbstractInterval}, μ)
    untruncated_d = likelihood_distribution(Z.Z, μ)
    _truncated(untruncated_d, Z.truncation_set) # TODO: introduce truncated subject to more general constraints
end

Base.@kwdef struct SelectionTilted{D<:Distributions.ContinuousUnivariateDistribution, F1,  T, EB, S} <: Distributions.ContinuousUnivariateDistribution
    untilted::D      # the original distribution
    tilting_function::F1
    selection_probability::T
    log_selection_probability::T
    truncation_sample::EB
    truncation_set::S
end

function Base.show(io::IO, d::SelectionTilted)
    print(io, "SelectionTilted{", string(d.untilted)," to ", string(d.truncation_sample),"}")
end
function tilt(d, Z)
    selection_probability = pdf(d, Z)
    log_selection_probability = logpdf(d, Z)
    tilting_function(μ) = Empirikos.likelihood(Z, μ)
    truncation_set = response(Z)
    SelectionTilted(;untilted = d,
        tilting_function = tilting_function,
        selection_probability = selection_probability,
        log_selection_probability = log_selection_probability,
        truncation_sample = Z,
        truncation_set = truncation_set
    )
end

function untilt(d::SelectionTilted)
    d.untilted
end

function selection_probability(d::SelectionTilted)
    d.selection_probability
end

function tilt(𝒢::AbstractMixturePriorClass, Z_trunc_set)
    MixturePriorClass(tilt.(components(𝒢), Z_trunc_set))
end


function Distributions.pdf(d::SelectionTilted, x::Real)
    Distributions.pdf(d.untilted, x) / d.selection_probability * d.tilting_function(x)
end

function marginalize(Z_trunc::TruncatedSample, prior::SelectionTilted)
    Z_untrunc = Z_trunc.Z
    truncation_set = Z_trunc.truncation_set
    if set_response(Z_trunc.Z, truncation_set) != prior.truncation_sample
        throw("selection tilt and truncated sample do not match")
    end
    marginal_untrunc = marginalize(Z_untrunc, prior.untilted)
    Distributions.truncated(marginal_untrunc, truncation_set)
end

struct ExtendedMarginalDensity{T} <: LinearEBayesTarget
    Z::T
end

location(target::ExtendedMarginalDensity) = target.Z


function (target::ExtendedMarginalDensity)(prior::SelectionTilted)
    # code duplication with marginalize.
    Z_trunc = target.Z
    Z_untrunc = Z_trunc.Z
    truncation_set = Z_trunc.truncation_set
    if set_response(Z_trunc.Z, truncation_set) != prior.truncation_sample
        throw("selection tilt and truncated sample do not match")
    end
    Distributions.pdf(prior.untilted, Z_untrunc) / prior.selection_probability
end


function (target::ExtendedMarginalDensity)(d::MixtureModel)
    sum( probs(d) .* target.(components(d)))
end


struct UntiltNormalizationConstant <: LinearEBayesTarget
end

(::UntiltNormalizationConstant)(d::SelectionTilted) = 1/selection_probability(d)
function (target::UntiltNormalizationConstant)(d::MixtureModel)
    sum(Distributions.probs(d) .*  target.(components(d)))
end

#
# probably need to say how we are pretilting to avoid bugs
# but for now let's assume pretilt is with respect to identical tilt of selection measure
struct UntiltLinearFunctionalNumerator{T} <: LinearEBayesTarget
    target::T
end

function (pretilt_target::UntiltLinearFunctionalNumerator)(d::SelectionTilted)
    target = pretilt_target.target
    target(untilt(d)) / selection_probability(d)
end

function (target::UntiltLinearFunctionalNumerator)(d::MixtureModel)
    sum(Distributions.probs(d) .*  target.(components(d)))
end


struct UntiltedLinearTarget{T<:LinearEBayesTarget} <: AbstractPosteriorTarget
    target::T
end

function untilt(target::LinearEBayesTarget)
    UntiltedLinearTarget(target)
end


Base.denominator(::UntiltedLinearTarget) = UntiltNormalizationConstant()
function Base.numerator(target::UntiltedLinearTarget)
    UntiltLinearFunctionalNumerator(target.target)
end

function (target::UntiltedLinearTarget)(d)
    numerator(target)(d) / denominator(target)(d)
end


struct UntiltedPosteriorTarget{T<:BasicPosteriorTarget} <: AbstractPosteriorTarget
    target::T
end

function untilt(target::BasicPosteriorTarget)
    UntiltedPosteriorTarget(target)
end



function (target::UntiltedPosteriorTarget)(d)
    numerator(target)(d) / denominator(target)(d)
end


function Base.denominator(target::UntiltedPosteriorTarget)
    Base.numerator(untilt(Base.denominator(target.target)))
end

function Base.numerator(target::UntiltedPosteriorTarget)
    Base.numerator(untilt(Base.numerator(target.target)))
end



#=
struct SelectionTilted{D<:Distributions.ContinuousUnivariateDistribution, EB, I,
         T<: Real, TL<:Union{T,Nothing}, TU<:Union{T,Nothing}} <: Distributions.ContinuousUnivariateDistribution
    untruncated::D      # the original distribution (untruncated)
    ebayes_sample::EB
    selection_event::I
    log_selection_probability::T

    lower::TL     # lower bound
    upper::TU     # upper bound
    loglcdf::T    # log-cdf of lower bound (exclusive): log P(X < lower)
    lcdf::T       # cdf of lower bound (exclusive): P(X < lower)
    ucdf::T       # cdf of upper bound (inclusive): P(X ≤ upper)

    tp::T         # the probability of the truncated part, i.e. ucdf - lcdf
    logtp::T      # log(tp), i.e. log(ucdf - lcdf)

    function Truncated(d::UnivariateDistribution, l::TL, u::TU, loglcdf::T, lcdf::T, ucdf::T, tp::T, logtp::T) where {T <: Real, TL <: Union{T,Nothing}, TU <: Union{T,Nothing}}
        new{typeof(d), value_support(typeof(d)), T, TL, TU}(d, l, u, loglcdf, lcdf, ucdf, tp, logtp)
    end
end =#
