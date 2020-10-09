#---------------------------------------
# types for a single EB sample
#---------------------------------------
abstract type EBayesSample{T} end

function likelihood_distribution end
function  response end
function  nuisance_parameter end

# trait
abstract type Skedasticity end
struct Homoskedastic <: Skedasticity end
struct Heteroskedastic <: Skedasticity end

function skedasticity(Zs::AbstractVector{EB}) where EB <: EBayesSample
    length(unique(nuisance_parameter.(Zs))) == 1 ? Homoskedastic() : Heteroskedastic()
end

# avoid piracy
likelihood(Z::EBayesSample, param) = _pdf(likelihood_distribution(Z, param), response(Z)) # maybe dispatch to Z first, then can deal with discretized sample
loglikelihood(Z::EBayesSample, param) = _logpdf(likelihood_distribution(Z, param), response(Z))

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

#isdiscretized() = ...


# Discretized{EBS, Interval}



"""
	marginalize(Z, prior)

Given a `prior` distribution ``G`` and  `EBayesSample` ``Z``,
return that marginal distribution of ``Z``.
"""
function marginalize end



# basic interface
# pdf(Z, NormalSample()) -> marginalization ...
# posterior(sample , prior) -> Distribution...
# marginal()

# target(Z)(prior) ->
# target(Z)(μ) = μ # dirac mass prior type of computation



#






# Poisson

# Binomial

# Replicated , ReplicatedArray
#
