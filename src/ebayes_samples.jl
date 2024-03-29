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

function set_response(Z::EBayesSample, znew=missing)
    Z = @set Z.Z = znew
end


function nuisance_parameter end

function Base.Float64(Z::EBayesSample{<:Number})
    Base.Float64(response(Z))
end

Base.isnan(Z::EBayesSample) = Base.isnan(response(Z))
Base.isfinite(Z::EBayesSample) = Base.isfinite(response(Z))


function Base.isless(a::EBayesSample, b::EBayesSample)
    Base.isless( (response(a), nuisance_parameter(a)),
                 (response(b), nuisance_parameter(b)) )
end

Base.isless(a::EBayesSample, b::Number) = Base.isless(response(a),b)
Base.isless(a::Number, b::EBayesSample) = Base.isless(a, response(b))


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
likelihood(Z::EBayesSample, param) = _pdf(likelihood_distribution(Z, param), response(Z))
loglikelihood(Z::EBayesSample, param) =
    _logpdf(likelihood_distribution(Z, param), response(Z))
loglikelihood(Z::EBayesSample, prior::Distribution) =
    _logpdf(marginalize(Z, prior), response(Z))


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
pdf(prior::Distribution, Z::EBayesSample) = _pdf(marginalize(Z, prior), response(Z))

"""
    cdf(prior::Distribution, Z::EBayesSample)

Given a `prior` ``G`` and `EBayesSample` ``Z``, evaluate the CDF of the marginal
distribution of ``Z`` at `response(Z)`.
"""
cdf(prior::Distribution, Z::EBayesSample) = _cdf(marginalize(Z, prior), response(Z))
"""
    ccdf(prior::Distribution, Z::EBayesSample)

Given a `prior` ``G`` and `EBayesSample` ``Z``, evaluate the complementary CDF of the marginal
distribution of ``Z`` at `response(Z)`.
"""
ccdf(prior::Distribution, Z::EBayesSample) = _ccdf(marginalize(Z, prior), response(Z))

logpdf(prior::Distribution, Z::EBayesSample) = _logpdf(marginalize(Z, prior), response(Z))




_pdf(dbn, z) = pdf(dbn, z)
_logpdf(dbn, z) = logpdf(dbn, z)

_cdf(dbn, z) = cdf(dbn, z)
_ccdf(dbn, z) = ccdf(dbn, z)
#const EBInterval{T} = Union{[Interval{T, S, R} for S in [Open, Closed, Unbounded], R in [Open, Closed, Unbounded]]...} where T

const EBInterval{T} = Union{
    Interval{T,Unbounded,Closed},
    Interval{T,Open,Unbounded},
    Interval{T,Open,Closed},
} where {T}

const EBIntervalFlipped{T} = Union{
    Interval{T,Closed,Open},
    Interval{T,Unbounded,Open},
    Interval{T,Closed,Unbounded},
} where {T}

function _cdf(dbn, interval::Interval{T,S,Unbounded}) where {T,S}
    one(eltype(dbn))
end

function _cdf(
    dbn::ContinuousUnivariateDistribution,
    interval::Interval{T,S,B},
) where {T,S,B<:Bounded}
    _cdf(dbn, last(interval))
end

#-- Unbounded - Unbounded
function _pdf(dbn, interval::Interval{T,Unbounded,Unbounded}) where {T}
    one(eltype(dbn))
end
function _logpdf(dbn, interval::Interval{T,Unbounded,Unbounded}) where {T}
    zero(eltype(dbn))
end

#-- Unbounded - Closed
function _pdf(dbn, interval::Interval{T,Unbounded,Closed}) where {T}
    cdf(dbn, last(interval))
end
function _logpdf(dbn, interval::Interval{T,Unbounded,Closed}) where {T}
    logcdf(dbn, last(interval))
end

#-- Open - Unbounded
function _pdf(dbn, interval::Interval{T,Open,Unbounded}) where {T}
    ccdf(dbn, first(interval))
end
function _logpdf(dbn, interval::Interval{T,Open,Unbounded}) where {T}
    logccdf(dbn, first(interval))
end

#-- Open - Closed
function _pdf(dbn, interval::Interval{T,Open,Closed}) where {T}
    cdf(dbn, last(interval)) - cdf(dbn, first(interval))
end
function _logpdf(dbn, interval::Interval{T,Open,Closed}) where {T}
    logdiffcdf(dbn, last(interval), first(interval))
end

# In the case of continuous distributions, we can delegate most results to  (Open, Closed)
for f in [:_pdf, :_logpdf]
    @eval begin
        # Closed - Closed and Closed - Open and Open - Open
        function $f(
            dbn::ContinuousUnivariateDistribution,
            interval::Union{
                Interval{T,Closed,Closed},
                Interval{T,Closed,Open},
                Interval{T,Open,Open},
            },
        ) where {T}
            _interval = Interval{T,Open,Closed}(first(interval), last(interval))
            $f(dbn, _interval)
        end
        # Unbounded - Open
        function $f(
            dbn::ContinuousUnivariateDistribution,
            interval::Interval{T,Unbounded,Open},
        ) where {T}
            _interval = Interval{T,Unbounded,Closed}(first(interval), last(interval))
            $f(dbn, _interval)
        end
        # Closed - Unbounded
        function $f(
            dbn::ContinuousUnivariateDistribution,
            interval::Interval{T,Closed,Unbounded},
        ) where {T}
            _interval = Interval{T,Open,Unbounded}(first(interval), last(interval))
            $f(dbn, _interval)
        end
    end
end

#TODO: Handle discrete distributions (less important)
function _pdf(dbn::DiscreteDistribution, interval::Interval{T,Closed,Closed}) where {T}
    cdf(dbn, last(interval)) - cdf(dbn, first(interval)) + pdf(dbn, first(interval))
end

function _cdf(
    dbn::DiscreteDistribution,
    interval::Interval{T,S,Closed},
) where {T,S}
    _cdf(dbn, last(interval))
end

function _pdf(dbn::DiscreteDistribution, interval::Interval{T,Closed,Unbounded}) where {T}
    ccdf(dbn, first(interval)) + pdf(dbn, first(interval))
end

function _pdf(dbn::Normal, interval::Interval{T,Closed,Unbounded}) where {T}
    if iszero(var(dbn))
        return _pdf(Dirac(mean(dbn)), interval)
    else
        return ccdf(dbn, first(interval))
    end
end

function _pdf(dbn::Normal, interval::Interval{T,Closed,Closed}) where {T}
    if iszero(var(dbn))
        return _pdf(Dirac(mean(dbn)), interval)
    else
        return  cdf(dbn, last(interval)) - cdf(dbn, first(interval))
    end
end



function _logpdf(dbn::DiscreteDistribution, interval::Interval{T,Closed,Unbounded}) where {T}
    log(_pdf(dbn, interval))
end


function _support(d::Distribution)
    distributions_interval_to_interval(support(d))
end

function distributions_interval_to_interval(interval::Distributions.RealInterval)
    _lb = isinf(interval.lb) ? nothing : interval.lb
    _ub = isinf(interval.ub) ? nothing : interval.ub
    Interval(_lb, _ub)
end



struct MultinomialSummary{T,S,D<:AbstractDict{T,S}}
    store::D #TODO, use other container
    effective_nobs::Int 
end

function MultinomialSummary(store::AbstractDict; effective_nobs = sum(values(store)))
    MultinomialSummary(store, effective_nobs)
end

# Is the below needed?
#function MultinomialSummary(vals, cnts)
#     MultinomialSummary(SortedDict(Dict(vals .=> cnts)))
#end


const VectorOrSummary{T} = Union{AbstractVector{T},MultinomialSummary{T}}

# Does the distinction of the two things below really make sense?
function (Zs_summary::MultinomialSummary)(Z)
    get(Zs_summary.store, Z, zero(Int))
end

function Base.getindex(Zs_summary::MultinomialSummary, i)
    Base.getindex(Zs_summary.store, i)
end

Base.keys(Zs_summary::MultinomialSummary) = Base.keys(Zs_summary.store)
Base.values(Zs_summary::MultinomialSummary) = Base.values(Zs_summary.store)
Base.length(Zs_summary::MultinomialSummary) = Base.length(Zs_summary.store)

# TODO: Move to Dictionaries.jl

function Base.broadcasted(::typeof(response), Zs_summary::MultinomialSummary)
    response.(keys(Zs_summary))
end

function Base.broadcasted(::typeof(pdf), prior, Zs_summary::MultinomialSummary)
    # Should this also return keys?
    pdf.(prior, collect(keys(Zs_summary.store)))
end

function Base.broadcasted(f, Zs_summary::MultinomialSummary)
    # Should this also return keys?
    f.(keys(Zs_summary.store))
end


function StatsBase.weights(Zs_summary::MultinomialSummary)
    fweights(collect(values(Zs_summary.store)))
end

summarize(Zs::AbstractVector) = MultinomialSummary(SortedDict(countmap(Zs)))
summarize(Zs::AbstractVector, ws::StatsBase.AbstractWeights) = MultinomialSummary(SortedDict(countmap(Zs, ws)))

function summarize(Zs::AbstractVector, ws::AbstractVector{Int}) 
    summarize(Zs, fweights(ws))
end

summarize(Zs::MultinomialSummary) = Zs


function skedasticity(Zs_summary::MultinomialSummary)
    all_unique_samples = collect(keys(Zs_summary.store))
    skedasticity(all_unique_samples)
end

function loglikelihood(mult::MultinomialSummary, prior)
    sum([n * loglikelihood(Z, prior) for (Z, n) in mult.store])
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
