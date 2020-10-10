#---------------------------------------
# types for a single EB sample
#---------------------------------------
abstract type EBayesSample{T} end

function likelihood_distribution end
function  response end
function  nuisance_parameter end

"""
	marginalize(Z, prior)

Given a `prior` distribution ``G`` and  `EBayesSample` ``Z``,
return that marginal distribution of ``Z``. Works for EBayesSample{Missing},
i.e., no realization is needed.
"""
function marginalize end

function posterior end

# trait
abstract type Skedasticity end
struct Homoskedastic <: Skedasticity end
struct Heteroskedastic <: Skedasticity end

function skedasticity(Zs::AbstractVector{EB}) where EB <: EBayesSample
    # probably should check that we have a concrete type..
    #Base.typename(typeof(BinomialSample(2,3)))
    length(unique(nuisance_parameter.(Zs))) == 1 ? Homoskedastic() : Heteroskedastic()
end

# avoid piracy
likelihood(Z::EBayesSample, param) = _pdf(likelihood_distribution(Z, param), response(Z)) # maybe dispatch to Z first, then can deal with discretized sample
loglikelihood(Z::EBayesSample, param) = _logpdf(likelihood_distribution(Z, param), response(Z))

loglikelihood(Z::EBayesSample, prior::Distribution) = logpdf(marginalize(Z, prior), response(Z))



_pdf(dbn, z) = pdf(dbn, z)
_logpdf(dbn, z) = logpdf(dbn, z)

#const EBInterval{T} = Union{[Interval{T, S, R} for S in [Open, Closed, Unbounded], R in [Open, Closed, Unbounded]]...} where T

const EBInterval{T} = Union{Interval{T, Unbounded, Unbounded},
                            Interval{T, Unbounded, Closed},
                            Interval{T, Open, Unbounded},
                            Interval{T, Open, Closed}} where T


function _pdf(dbn, interval::Interval{T, Unbounded, Unbounded}) where T
    one(eltype(interval))
end

function _logpdf(dbn, interval::Interval{T, Unbounded, Unbounded}) where T
    zero(eltype(interval))
end

# need better handling if (Unbounded, Open)
function _pdf(dbn, interval::Interval{T, Unbounded, Closed}) where T
    cdf(dbn, last(interval))
end

function _logpdf(dbn, interval::Interval{T, Unbounded, Closed}) where T
    logcdf(dbn, last(interval))
end

function _pdf(dbn, interval::Interval{T, Open, Unbounded}) where T
    ccdf(dbn, first(interval))
end

function _logpdf(dbn, interval::Interval{T, Open, Unbounded}) where T
    logccdf(dbn, first(interval))
end

function _pdf(dbn, interval::Interval{T, Open, Closed}) where T
    cdf(dbn, last(interval)) - cdf(dbn, first(interval))
end

function _logpdf(dbn, interval::Interval{T, Open, Closed}) where T
    logdiffcdf(dbn, last(interval), first(interval))
end



struct MultinomialSummary{T}
    store::Dict{T,Int} #TODO, use other container
end

summarize(Zs::AbstractVector{<:EBayesSample}) = MultinomialSummary(countmap, Zs)

function skedasticity(Zs_summary::MultinomialSummary)
    all_unique_samples = collect(keys(Zs_summary.store))
    skedasticity(all_unique_samples)
end


function loglikelihood(mult::MultinomialSummary, prior)
    sum([n*loglikelihood(Z, prior) for (Z,n) in mult.store])
end






# target(Z)(prior) ->
# target(Z)(μ) = μ # dirac mass prior type of computation



#  Mixture of Conjugates functionality



#function marginalize(MixtureModel{}, Z::EBayesSample)
#
#end


#function posterior(MixtureModel{}, Z::EBayesSample)
#
#end



# Poisson
