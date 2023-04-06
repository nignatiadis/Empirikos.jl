# Some additional checks compared to
# test_lord_cressie.jl

using Empirikos
using Test
using Hypatia
using Random
using StableRNGs


rng = StableRNG(1)
Zs_standard = StandardNormalSample.(randn(rng, 1000) .* sqrt(2))
true_prior = Normal(0, 1)

g_prior = MixturePriorClass( Normal.(-2:0.1:2, 1))

gauss_loc_02 = InfinityNormDensityBand(; a_min = -3.0, a_max = 3.0, α = 0.2, rng=StableRNG(100))

# deepcopying below to make sure RNG status remains the same.
fitted_gauss_loc_02 = fit(gauss_loc_02, Zs_standard)

floc_interval = FLocalizationInterval(;flocalization = gauss_loc_02,
                                       solver = Hypatia.Optimizer,
                                       convexclass = g_prior)


floc_interval_prefitted = FLocalizationInterval(;flocalization = fitted_gauss_loc_02,
                                       solver = Hypatia.Optimizer,
                                       convexclass = g_prior)

target1 = PosteriorMean(StandardNormalSample(1.0))

@test target1(true_prior) == 0.5

# check if result is same with unfitted or prefitted flocalization

confint1 = confint(floc_interval, target1, Zs_standard)
confint1_prefitted = confint(floc_interval_prefitted, target1, Zs_standard)
confint1_prefitted2 = confint(floc_interval_prefitted, target1)

# Compare to mosek solution
mosek_confint1_lower = 0.45212135895206856
mosek_confint1_upper = 0.6648705613789273
@test confint1.lower ≈ mosek_confint1_lower rtol=0.005
@test confint1.upper ≈ mosek_confint1_upper rtol=0.005

@test confint1 == confint1_prefitted
@test confint1 == confint1_prefitted2


# For KDE-F-Localization, check whether solutions with LinearVexity and ConvexVexity match up to numerical error.


confint1_prefitted_convexvexity = confint(floc_interval_prefitted, target1,
                                        Zs_standard, Empirikos.ConvexVexity())


@test confint1.α == confint1_prefitted_convexvexity.α
@test confint1.lower ≈ confint1_prefitted_convexvexity.lower rtol=0.005
@test confint1.upper ≈ confint1_prefitted_convexvexity.upper rtol=0.005

# For multiple targets, check if broadcasting works correctly.

target2 = PosteriorMean(StandardNormalSample(2.0))
@test target2(true_prior) == 1.0

confint2 = confint(floc_interval, target2, Zs_standard)

both_targets = [target1; target2]

both_cis = confint.(floc_interval, both_targets, Zs_standard)
@test both_cis[1] == confint1
@test both_cis[2] == confint2

both_cis_prefitted = confint.(floc_interval_prefitted, both_targets)
@test both_cis_prefitted == both_cis

both_cis_level = confint.(floc_interval, both_targets, Zs_standard; level=0.8)
@test both_cis == both_cis_level

both_cis_prefitted_level = confint.(floc_interval_prefitted, both_targets; level=0.8)
@test both_cis_prefitted_level == both_cis

# Check if level is double-checked when using `confint` command.
@test_throws ErrorException confint(floc_interval, target1; level = 0.95)
@test_throws ErrorException confint.(floc_interval, both_targets, Zs_standard; level=0.95)
@test_throws ErrorException confint.(floc_interval_prefitted, both_targets; level=0.1)

# When target is marginal density, check whether it works for linear functionals.

density_target = MarginalDensity(StandardNormalSample(0.0))
estimated_density_at_1 = fitted_gauss_loc_02.interp_kde.itp(0.0)
band_width = fitted_gauss_loc_02.C∞
true_density_at_1 = density_target(true_prior)

infinity_band_ub = estimated_density_at_1 + band_width
infinity_band_lb = estimated_density_at_1 - band_width

density_ci = confint(floc_interval_prefitted, density_target)

@test infinity_band_ub >= density_ci.upper
@test infinity_band_lb <= density_ci.lower * 1.000001


# now use a more-fine grained prior class so that we can get exact equality
floc_interval_discrete = FLocalizationInterval(;flocalization = fitted_gauss_loc_02,
                                       solver = Hypatia.Optimizer,
                                       convexclass = DiscretePriorClass(-3:0.01:3))

density_ci_discrete = confint(floc_interval_discrete, density_target)

@test infinity_band_ub ≈ density_ci_discrete.upper rtol=0.001
@test infinity_band_lb ≈ density_ci_discrete.lower rtol=0.001


# Intervals are nested for larger level.

gauss_loc_01 = InfinityNormDensityBand(; a_min = -3.0, a_max = 3.0, α = 0.1)

floc_interval_01 = FLocalizationInterval(;flocalization = gauss_loc_01,
                                       solver = Hypatia.Optimizer,
                                       convexclass = g_prior)
confint1_01 = confint(floc_interval_01, target1, Zs_standard)

@test confint1_01.α == 0.1
@test confint1_01.lower < confint1.lower
@test confint1_01.upper > confint1.upper
