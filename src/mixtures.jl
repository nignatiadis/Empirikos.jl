#  Mixture of Conjugates functionality


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
    πs = probs(prior)
    fs = likelihood.(Z, support(prior))
    posterior_πs = πs .* fs
    posterior_πs /= sum(posterior_πs)
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
    πs = probs(prior)
    fs = pdf.(components(prior), Z)
    posterior_πs = πs .* fs
    posterior_πs /= sum(posterior_πs)
    posterior_components = posterior.(Z, components(prior))
    MixtureModel(posterior_components, posterior_πs)
end
