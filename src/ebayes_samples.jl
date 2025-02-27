#---------------------------------------
# types for a single EB sample
#---------------------------------------

"""
    EBayesSample{T}

Abstract type representing empirical Bayes samples with realizations of type `T`.
"""
abstract type EBayesSample{T} end

abstract type ContinuousEBayesSample{T} <: EBayesSample{T} end
abstract type DiscreteEBayesSample{T} <: EBayesSample{T} end


"""
    likelihood_distribution(Z::EBayesSample, μ::Number)

Returns the distribution ``p(\\cdot \\mid \\mu)`` of ``Z \\mid \\mu`` (the return type being
a `Distributions.jl` Distribution).

# Examples
```jldoctest
julia> likelihood_distribution(StandardNormalSample(1.0), 2.0)
Normal{Float64}(μ=2.0, σ=1.0)
```
"""
function likelihood_distribution end

"""
    response(Z::EBayesSample{T})

Returns the concrete realization of `Z` as type `T`, thus dropping the information about the
likelihood.

# Examples
```jldoctest
julia> response(StandardNormalSample(1.0))
1.0
```
"""
function response(Z::EBayesSample)
   Z.Z
end

StatsDiscretizations.unwrap(Z::EBayesSample) = response(Z)

function set_response(Z::EBayesSample, znew=missing)
    Z = @set Z.Z = znew
end

StatsDiscretizations.wrap(Z::EBayesSample, val) = set_response(Z, val)


function nuisance_parameter end

function Base.Float64(Z::EBayesSample{<:Number})
    Base.Float64(response(Z))
end

Base.isnan(Z::EBayesSample) = Base.isnan(response(Z))
Base.isfinite(Z::EBayesSample) = Base.isfinite(response(Z))



"""
	marginalize(Z::EBayesSample, prior::Distribution)

Given a `prior` distribution ``G`` and  `EBayesSample` ``Z``,
return that marginal distribution of ``Z``. Works for `EBayesSample{Missing}``,
i.e., no realization is needed.

# Examples
```jldoctest
julia> marginalize(StandardNormalSample(1.0), Normal(2.0, sqrt(3)))
Normal{Float64}(μ=2.0, σ=1.9999999999999998)
````
"""
function marginalize end

function posterior end

Broadcast.broadcastable(Z::EBayesSample) = Ref(Z)

function multiplicity(Zs)
    collect(StatsBase.weights(Zs))
end
function StatsBase.weights(Zs::AbstractVector{<:EBayesSample})
    uweights(length(Zs))
end

summarize_by_default(Zs) = false


# trait
abstract type Skedasticity end
struct Homoskedastic <: Skedasticity end
struct Heteroskedastic <: Skedasticity end

function skedasticity(Zs::AbstractVector{EB}) where {EB<:EBayesSample}
    # probably should check that we have a concrete type..
    #Base.typename(typeof(BinomialSample(2,3)))
    length(unique(nuisance_parameter.(Zs))) == 1 ? Homoskedastic() : Heteroskedastic()
end

# avoid piracy
likelihood(Z::EBayesSample, param) = StatsDiscretizations.pdf(likelihood_distribution(Z, param), response(Z))
loglikelihood(Z::EBayesSample, param) =
    StatsDiscretizations.logpdf(likelihood_distribution(Z, param), response(Z))
loglikelihood(Z::EBayesSample, prior::Distribution) =
    StatsDiscretizations.logpdf(marginalize(Z, prior), response(Z))


function loglikelihood(Zs::AbstractVector{<:EBayesSample}, prior)
    sum(loglikelihood.(Zs, prior))
end

"""
    pdf(prior::Distribution, Z::EBayesSample)

Given a `prior` ``G`` and `EBayesSample` ``Z``, compute the marginal density of `Z`.

# Examples
```jldoctest
julia> Z = StandardNormalSample(1.0)
N(1.0; μ, σ=1.0)
julia> prior = Normal(2.0, sqrt(3))
Normal{Float64}(μ=2.0, σ=1.7320508075688772)
julia> pdf(prior, Z)
0.17603266338214976
julia> pdf(Normal(2.0, 2.0), 1.0)
0.17603266338214976
```
"""
pdf(prior::Distribution, Z::EBayesSample) = StatsDiscretizations.pdf(marginalize(Z, prior), response(Z))

"""
    cdf(prior::Distribution, Z::EBayesSample)

Given a `prior` ``G`` and `EBayesSample` ``Z``, evaluate the CDF of the marginal
distribution of ``Z`` at `response(Z)`.
"""
cdf(prior::Distribution, Z::EBayesSample) = StatsDiscretizations.cdf(marginalize(Z, prior), response(Z))
"""
    ccdf(prior::Distribution, Z::EBayesSample)

Given a `prior` ``G`` and `EBayesSample` ``Z``, evaluate the complementary CDF of the marginal
distribution of ``Z`` at `response(Z)`.
"""
ccdf(prior::Distribution, Z::EBayesSample) = StatsDiscretizations.ccdf(marginalize(Z, prior), response(Z))

logpdf(prior::Distribution, Z::EBayesSample) = StatsDiscretizations.logpdf(marginalize(Z, prior), response(Z))

logcdf(prior::Distribution, Z::EBayesSample) = StatsDiscretizations.logcdf(marginalize(Z, prior), response(Z))


function _support(d::Distribution)
    distributions_interval_to_interval(support(d))
end

function distributions_interval_to_interval(interval::Distributions.RealInterval)
    _lb = interval.lb
    _ub = interval.ub
    Interval(_lb, _ub)
end





struct MultinomialSummary{T,D}
    store::D
    effective_nobs::Int 
end

function MultinomialSummary(store, effective_nobs)
    MultinomialSummary{keytype(store), typeof(store)}(store, effective_nobs)
end

function Base.show(io::IO, Z::MultinomialSummary)
    Base.show(io, Z.store)
end

const VectorOrSummary{T} = Union{AbstractVector{T},MultinomialSummary{T}}

# Does the distinction of the two things below really make sense?
function (Zs_summary::MultinomialSummary)(Z)
    get(Zs_summary.store, Z, zero(Int))
end

function Base.getindex(Zs_summary::MultinomialSummary, i)
    Base.getindex(collect(Base.keys(Zs_summary.store)), i)
end

Base.keys(Zs_summary::MultinomialSummary) = Base.keys(Zs_summary.store)
Base.values(Zs_summary::MultinomialSummary) = Base.values(Zs_summary.store)
Base.length(Zs_summary::MultinomialSummary) = Base.length(Zs_summary.store)

# TODO: Move to Dictionaries.jl

function Base.broadcasted(::typeof(response), Zs_summary::MultinomialSummary)
    response.(keys(Zs_summary))
end

function Base.broadcasted(f, prior, Zs_summary::MultinomialSummary)
    # Should this also return keys?
    f.(prior, collect(keys(Zs_summary.store)))
end

function Base.broadcasted(f, Zs_summary::MultinomialSummary)
    # Should this also return keys?
    f.(keys(Zs_summary.store))
end


function StatsBase.weights(Zs_summary::MultinomialSummary)
    fweights(collect(values(Zs_summary.store)))
end

function summarize(args...; effective_nobs=nothing)
    _dict = StatsDiscretizations.sorted_countmap(args...)
    if isnothing(effective_nobs)
        effective_nobs = sum(values(_dict))
    end
    MultinomialSummary(_dict, effective_nobs)
end

summarize(Zs::MultinomialSummary) = Zs

function skedasticity(Zs_summary::MultinomialSummary)
    all_unique_samples = collect(keys(Zs_summary.store))
    skedasticity(all_unique_samples)
end

function loglikelihood(mult::MultinomialSummary, prior)
    sum(mult.store .* loglikelihood.(keys(mult.store), prior))
end


nobs(Zs_summary::MultinomialSummary) = Zs_summary.effective_nobs
nobs(Zs::AbstractVector{<:EBayesSample}) = length(Zs)


StatsBase.fit(::Nothing, ::VectorOrSummary) = nothing






# Recall difference between truncating marginal VS truncated likelihood
# Truncated(Normal())
#------------------------------------------
# MarginalTruncated{...} and Truncated{...}
#------------------------------------------
# w Dirac prior these are the same!
