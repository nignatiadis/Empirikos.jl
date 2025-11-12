using Empirikos
using Test 
using Distributions

# Below are numerically unstable codes we used previously.

function tmp_posterior(Z::EBayesSample, prior::DiscreteNonParametric)
    πs = probs(prior)
    fs = likelihood.(Z, support(prior))
    posterior_πs = πs .* fs
    posterior_πs /= sum(posterior_πs)
    DiscreteNonParametric(support(prior), posterior_πs)
end

function tmp_posterior(Z::EBayesSample, prior::MixtureModel)
    πs = probs(prior)
    fs = pdf.(components(prior), Z)
    posterior_πs = πs .* fs
    posterior_πs /= sum(posterior_πs)
    posterior_components = Empirikos.posterior.(Z, components(prior))
    MixtureModel(posterior_components, posterior_πs)
end

G_discr = DiscreteNonParametric([1.0, 3.0], [0.3, 0.7])
G_normalmix = MixtureModel([Normal(1.0, 0.5), Normal(3.0, 0.5)], [0.4, 0.6])
Z = StandardNormalSample(100.0)

@test_throws DomainError tmp_posterior(Z, G_discr)
@test_throws DomainError tmp_posterior(Z, G_normalmix)

post_discr = Empirikos.posterior(Z, G_discr)
@test probs(post_discr) ≈ [0.0,1.0]
@test support(post_discr) == support(G_discr)

post_normalmix = Empirikos.posterior(Z, G_normalmix)
@test probs(post_normalmix) ≈ [0.0,1.0]
@test components(post_normalmix)[2] == Empirikos.posterior(Z, components(G_normalmix)[2])
@test components(post_normalmix)[1] == Empirikos.posterior(Z, components(G_normalmix)[1])






