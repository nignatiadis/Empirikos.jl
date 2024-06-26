using Empirikos
using Test
using StableRNGs
using Hypatia

Z = BinomialSample(10)



betabin = marginalize(Z, Beta(2,1))

Zs = BinomialSample.(rand(StableRNG(1), betabin, 100), 10);
unique_Zs = sort(unique(Zs), by=response)
Zs_summary = summarize(Zs)


cvx_class = DiscretePriorClass(0.01:0.01:0.99)
npmle = NPMLE(cvx_class, Hypatia.Optimizer)
npmle_fit = fit(NPMLE(cvx_class, Hypatia.Optimizer), Zs)
npmle_fit_summary = fit(NPMLE(cvx_class, Hypatia.Optimizer), Zs_summary)

#using MosekTools
#npmle_mosek_fit = fit(NPMLE(cvx_class, Mosek.Optimizer), Zs)
#loglikelihood(Zs, npmle_mosek_fit.prior)

mosek_loglikelihood = -220.2670724305244
# figure out why tolerance has changed
@test loglikelihood(Zs, npmle_fit_summary.prior) ≈ mosek_loglikelihood atol = 1e-3
@test loglikelihood(Zs, npmle_fit_summary.prior) ≈ loglikelihood(Zs, npmle_fit.prior) atol = 1e-3
@test loglikelihood(Zs, npmle_fit_summary.prior) ≈ loglikelihood(Zs_summary, npmle_fit_summary.prior) atol = 1e-3
@test loglikelihood(Zs, npmle_fit.prior) ≈ loglikelihood(Zs_summary, npmle_fit.prior) atol = 1e-3



@test probs(npmle_fit_summary.prior) ≈ probs(npmle_fit.prior) atol=1e-3

estimated_marginal_probs = pdf.(npmle_fit, unique_Zs)
true_marginal_probs = pdf.(betabin, response.(unique_Zs))

#plot(support(npmle_fit.prior), npmle_fit.prior, seriestype=:sticks)
#plot!(Beta(2,1))
#plot(0:10, true_marginal_probs, seriestype=:sticks)
#plot!(0:10, estimated_marginal_probs,  seriestype=:scatter)

marginalize.(unique_Zs, npmle_fit)

unique_binomial_samples = BinomialSample.(0:12, 12)
unique_binomial_counts = [3; 24; 104; 286; 670; 1033; 1343; 1112; 829; 478; 181; 45; 7]

mult_binom = Empirikos.summarize(unique_binomial_samples, unique_binomial_counts)


#beta_mle = fit(ParametricMLE(Beta()), mult_binom)

α_wiki = 34.1350
β_wiki = 31.6085
likelihood_wiki = -12492.9

#@test beta_mle.α ≈ α_wiki atol = 0.05
#@test beta_mle.β ≈ β_wiki atol = 0.05
#@test loglikelihood(mult_binom, beta_mle) ≈ likelihood_wiki atol = 0.05
@test loglikelihood(mult_binom, Beta(α_wiki, β_wiki)) ≈ likelihood_wiki atol = 0.05
