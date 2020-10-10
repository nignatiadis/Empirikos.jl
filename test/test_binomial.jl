using Empirikos
using Test

Z = BinomialSample(10)

betabin = marginalize(Z, Beta(2,1))

Zs = BinomialSample.(rand(betabin, 10000), 10);



unique_binomial_samples = BinomialSample.(0:12, 12)
unique_binomial_counts = [3; 24; 104; 286; 670; 1033; 1343; 1112; 829; 478; 181; 45; 7]

dict_summary = Dict(unique_binomial_samples .=> unique_binomial_counts)

mult_binom = Empirikos.MultinomialSummary(dict_summary)
sort(collect(keys(mult_binom.store)))


beta_mle = fit(ParametricMLE(Beta()), mult_binom)

α_wiki = 34.1350
β_wiki = 31.6085
likelihood_wiki = -12492.9

@test beta_mle.α ≈ α_wiki atol = 0.05
@test beta_mle.β ≈ β_wiki atol = 0.05
@test loglikelihood(mult_binom, beta_mle) ≈ likelihood_wiki atol = 0.05
