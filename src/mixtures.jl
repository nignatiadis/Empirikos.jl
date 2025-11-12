#  Mixture of Conjugates functionality
struct ContinuousDirac{T} <: ContinuousUnivariateDistribution
    value::T
end

Distributions.Dirac(d::ContinuousDirac) = Dirac(d.value)
Distributions.insupport(d::ContinuousDirac, x::Real) = insupport(Dirac(d), x)
Distributions.pdf(d::ContinuousDirac, x::Real) = insupport(d, x) ? 1.0 : 0.0
Distributions.logpdf(d::ContinuousDirac, x::Real) = insupport(d, x) ? 0.0 : -Inf
Distributions.cdf(d::ContinuousDirac, x::Real) = x < d.value ? 0.0 : isnan(x) ? NaN : 1.0
Distributions.logcdf(d::ContinuousDirac, x::Real) = x < d.value ? -Inf : isnan(x) ? NaN : 0.0
Distributions.ccdf(d::ContinuousDirac, x::Real) = x < d.value ? 1.0 : isnan(x) ? NaN : 0.0
Distributions.logccdf(d::ContinuousDirac, x::Real) = x < d.value ? 0.0 : isnan(x) ? NaN : -Inf


const GDirac{T} = Union{ContinuousDirac{T}, Dirac{T}}

function default_target_computation(::BasicPosteriorTarget,
    ::EBayesSample,
    ::Dirac
)
    Conjugate()
end

function marginalize(Z::EBayesSample, G::GDirac)
    likelihood_distribution(Z, G.value)
end

function posterior(::EBayesSample, G::GDirac)
    G
end

const BivariateDirac{T} = Distributions.ProductDistribution{1,0, Tuple{Dirac{T}, Dirac{T}}}
const BivariateContinuousDirac{T} = Distributions.ProductDistribution{1,0, Tuple{ContinuousDirac{T}, ContinuousDirac{T}}}
const BivariateGDirac{T} = Union{BivariateDirac{T},  BivariateContinuousDirac{T}}


function default_target_computation(::BasicPosteriorTarget,
    ::EBayesSample,
    ::BivariateGDirac
)
    Conjugate()
end

#function posterior(::EBayesSample, G::BivariateGDirac)
#    G
#end



function default_target_computation(
    ::BasicPosteriorTarget,
    ::EBayesSample,
    ::DiscreteNonParametric
)
    Conjugate()
end



function marginalize(Z::EBayesSample, prior::DiscreteNonParametric)
    πs = probs(prior)
    distributions = likelihood_distribution.(Z, support(prior))
    #TODO: note that e.g. for Binomial this could directly return another DiscreteNonParametric.
    # In most cases this would not be worth the initial conversion cost?
    MixtureModel(distributions, πs)
end

function posterior(Z::EBayesSample, prior::DiscreteNonParametric)
    log_πs = log.(probs(prior))
    log_fs = loglikelihood.(Z, support(prior))
    log_posterior_πs = log_πs .+ log_fs
    log_normalizer = logsumexp(log_posterior_πs)
    posterior_πs = exp.(log_posterior_πs .- log_normalizer)
    DiscreteNonParametric(support(prior), posterior_πs)
end

function default_target_computation(
    ::BasicPosteriorTarget,
    ::EBayesSample,
    ::MixtureModel
)
    #TODO NOT always true.
    Conjugate()
end


function marginalize(Z::EBayesSample, prior::MixtureModel)
    πs = probs(prior)
    marginal_components = marginalize.(Z, components(prior))
    MixtureModel(marginal_components, πs)
end

function posterior(Z::EBayesSample, prior::MixtureModel)
    log_πs = log.(probs(prior))
    log_fs = logpdf.(components(prior), Z)
    log_posterior_πs = log_πs .+ log_fs
    log_normalizer = logsumexp(log_posterior_πs)
    posterior_πs = exp.(log_posterior_πs .- log_normalizer)
    posterior_components = posterior.(Z, components(prior))
    MixtureModel(posterior_components, posterior_πs)
end


