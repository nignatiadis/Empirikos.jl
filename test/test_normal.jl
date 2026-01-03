using Empirikos
using Test
using QuadGK
using Distributions

_target = MarginalDensity(StandardNormalSample(3.0))
@test Base.extrema(_target)[2] == pdf(Normal(),0)

@test Base.extrema(MarginalDensity(NormalSample(3.0, 0.5)))[2] == pdf(Normal(0,0.5),0.0)

n = Normal()
a = 1.0
b = 2.3

# Check if marginalization of a normal with an AffineDistribution is correct

prior = a + b * n
prior_locscale = Distributions.AffineDistribution(a, b, n)
prior_locscale_mixture = Distributions.AffineDistribution(a, b, MixtureModel([n],[1.0]))

Z = NormalSample(3.0, 0.5)
@test marginalize(Z, prior_locscale) == marginalize(Z, prior)

marg_mix = marginalize(Z, prior_locscale_mixture)
@test probs(marg_mix.ρ) == [1.0]
@test marg_mix.μ + marg_mix.σ * first(components(marg_mix.ρ)) == marginalize(Z, prior)


# Normal Uniform

u = Uniform(1.0, 4.0)
Z = NormalSample(2.0, 1.3)

post_Z = Empirikos.posterior(Z, u)

post_mean_1 = mean(post_Z)
post_mean_target = Empirikos.PosteriorMean(Z)
@test post_mean_1 ≈ post_mean_target(u)

post_mean_numerator = quadgk(μ-> μ * Empirikos.likelihood(Z,μ) * pdf(u,μ), u.a, u.b)[1]
post_mean_denominator = quadgk(μ-> Empirikos.likelihood(Z,μ) * pdf(u,μ), u.a, u.b)[1]
@test post_mean_1 ≈ post_mean_numerator / post_mean_denominator

@test pdf(u,Z) ≈ post_mean_denominator

marg_u = Empirikos.marginalize(Z, u)
@test pdf(marg_u, response(Z)) ≈ post_mean_denominator