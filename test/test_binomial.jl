using Empirikos
using Test
using StatsPlots
Z = BinomialSample(10)

betabin = marginalize(Z, Beta(2,1))

Zs = BinomialSample.(rand(betabin, 1000), 10);




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





Zs = BinomialSample.(rand(betabin, 200), 10);

histogram(response.(Zs), nbins=20)

using MosekTools
using MathOptInterface
using JuMP

model = Model(Mosek.Optimizer)
cvx_class = DiscretePriorClass(0.01:0.01:0.99)
π = Empirikos.prior_variable!(model, cvx_class)
f = pdf.(π, Zs)

@variable(model, u)
@variable(model, fmarg[1:200])
@constraint(model, fmarg .== f)

bla = vcat(u, fmarg, fill(1.0,200))
@constraint(model, bla in MathOptInterface.RelativeEntropyCone(401))
@objective(model, Min, u)
optimize!(model)

using Plots
function fix_πs(πs)
    πs = max.(πs, 0.0)
    πs = πs ./ sum(πs)
end

bla = cvx_class(fix_πs(JuMP.value.(π.finite_param)))


mymarginal = marginalize(Z, bla)

plot(mymarginal; components=false, seriestype=:sticks)

plot(support(bla), probs(bla), seriestype=:sticks)



using DataStructures


bla = summarize(Zs)
dkw_fit = fit(DvoretzkyKieferWolfowitz(0.05), bla)
dkw_fit.summary
expected_ub = dkw_fit.band  + dkw_fit.summary[BinomialSample(0,10)]

model = Model(Mosek.Optimizer)
cvx_class = DiscretePriorClass(0.01:0.01:0.99)
π = Empirikos.prior_variable!(model, cvx_class)
prob1 = MarginalDensity(BinomialSample(0,10))
prob1eval = prob1(π)

Empirikos.neighborhood_constraint!(model, dkw_fit, π)

@objective(model, Max, prob1eval)
optimize!(model)


@test expected_ub == JuMP.value(prob1eval)
plot(cvx_class.support, JuMP.value.(π.finite_param), seriestype=:sticks)
using Plots
f = pdf.(π, Zs)

@variable(model, u)


FittedDw

nobs(bla)



SortedDict(bla.store)



plot(1:10,1:10, series=:scatter)
#plot(Beta(2,1))


bla2 = MixtureModel([Normal(), Normal(2.0,2.0)], [0.4;0.6])

isa(bla2, ContinuousUnivariateDistribution)

bub = quantile.(components(bla2), 0.9)

Distributions.quantile_bisect(bla2, 0.9, bub[1], bub[2], 1.0e-12)


bla3 = MixtureModel([Normal(), Normal(2.0,2.0)], [1.0;0.0])
bub3 = quantile.(components(bla3), 0.9)

Distributions.quantile_bisect(bla3, 0.9, bub3[1], bub3[2], 1.0e-12)

bla4 = MixtureModel([Normal(), Laplace()], [1.0;0.0])
bub4 = quantile.(components(bla4), 0.9)

Distributions.quantile_bisect(bla4, 0.9, bub4[1], bub4[2], 1.0e-12)


blabla = MixtureModel([Poisson(), Poisson(2)])

#quantile(blabla, 3)

minimum(bla2)
isa(bla2, MixtureModel{Univariate, Continuous})


bla3 = MixtureModel([Normal(), Poisson()])
