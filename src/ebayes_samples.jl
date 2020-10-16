#---------------------------------------
# types for a single EB sample
#---------------------------------------
abstract type EBayesSample{T} end

function likelihood_distribution end
function response end
function nuisance_parameter end

"""
	marginalize(Z, prior)

Given a `prior` distribution ``G`` and  `EBayesSample` ``Z``,
return that marginal distribution of ``Z``. Works for EBayesSample{Missing},
i.e., no realization is needed.
"""
function marginalize end

function posterior end

Broadcast.broadcastable(Z::EBayesSample) = Ref(Z)

function multiplicity(Zs::AbstractVector{<:EBayesSample})
    fill(1, length(Zs))
end

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
likelihood(Z::EBayesSample, param) = _pdf(likelihood_distribution(Z, param), response(Z)) # maybe dispatch to Z first, then can deal with discretized sample
loglikelihood(Z::EBayesSample, param) =
    _logpdf(likelihood_distribution(Z, param), response(Z))
loglikelihood(Z::EBayesSample, prior::Distribution) =
    logpdf(marginalize(Z, prior), response(Z))


function loglikelihood(Zs::AbstractVector{<:EBayesSample}, prior)
    sum(loglikelihood.(Zs, prior))
end

pdf(prior, Z::EBayesSample) = _pdf(marginalize(Z, prior), response(Z))
cdf(prior, Z::EBayesSample) = cdf(marginalize(Z, prior), response(Z)) # Turn this also into _cdf eventually.
ccdf(prior, Z::EBayesSample) = ccdf(marginalize(Z, prior), response(Z))






_pdf(dbn, z) = pdf(dbn, z)
_logpdf(dbn, z) = logpdf(dbn, z)

#const EBInterval{T} = Union{[Interval{T, S, R} for S in [Open, Closed, Unbounded], R in [Open, Closed, Unbounded]]...} where T

const EBInterval{T} = Union{
    Interval{T,Unbounded,Unbounded},
    Interval{T,Unbounded,Closed},
    Interval{T,Open,Unbounded},
    Interval{T,Open,Closed},
} where {T}

const EBIntervalFlipped{T} = Union{
    Interval{T,Unbounded,Unbounded},
    Interval{T,Closed,Open},
    Interval{T,Unbounded,Open},
    Interval{T,Closed,Unbounded},
} where {T}


#-- Unbounded - Unbounded
function _pdf(dbn, interval::Interval{T,Unbounded,Unbounded}) where {T}
    one(eltype(interval))
end
function _logpdf(dbn, interval::Interval{T,Unbounded,Unbounded}) where {T}
    zero(eltype(interval))
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

# In the case of continuous distributions, we can delegate most result to  (Open, Closed)
for f in [:_pdf, :_logpdf]
    @eval begin
        # Closed - Closed and Closed - Open and Open - Open
        function $f(dbn::ContinuousUnivariateDistribution,
                    interval::Union{Interval{T,Closed,Closed}, Interval{T,Closed,Open}, Interval{T,Open,Open}}) where {T}
            _interval = Interval{T, Open, Closed}(first(interval), last(interval))
            $f(dbn, _interval)
        end
        # Unbounded - Open
        function $f(dbn::ContinuousUnivariateDistribution, interval::Interval{T,Unbounded,Open}) where {T}
            _interval = Interval{T, Unbounded, Closed}(first(interval), last(interval))
            $f(dbn, _interval)
        end
        # Closed - Unbounded
        function $f(dbn::ContinuousUnivariateDistribution, interval::Interval{T,Closed,Unbounded}) where {T}
            _interval = Interval{T, Open, Unbounded}(first(interval), last(interval))
            $f(dbn, _interval)
        end
    end
end

#TODO: Handle discrete distributions (less important)


struct MultinomialSummary{T,D<:AbstractDict{T,Int}}
    store::D #TODO, use other container
end

function Base.broadcasted(::typeof(pdf), prior, Zs_summary::MultinomialSummary)
    # Should this also return keys?
    pdf.(prior, collect(keys(Zs_summary.store)))
end



function multiplicity(Zs_summary::MultinomialSummary)
    collect(values(Zs_summary.store))
end


summarize(Zs::AbstractVector{<:EBayesSample}) = MultinomialSummary(SortedDict(countmap(Zs)))

function skedasticity(Zs_summary::MultinomialSummary)
    all_unique_samples = collect(keys(Zs_summary.store))
    skedasticity(all_unique_samples)
end

function loglikelihood(mult::MultinomialSummary, prior)
    sum([n * loglikelihood(Z, prior) for (Z, n) in mult.store])
end

nobs(Zs_summary::MultinomialSummary) = sum(values(Zs_summary.store))





# target(Z)(prior) ->
# target(Z)(μ) = μ # dirac mass prior type of computation


#function marginalize(MixtureModel{}, Z::EBayesSample)
#
#end


#function posterior(MixtureModel{}, Z::EBayesSample)
#
#end



# Poisson
