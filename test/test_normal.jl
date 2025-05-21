using Empirikos
using Test


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