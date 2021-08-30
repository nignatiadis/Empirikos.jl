# Bernoulli is good sanity check since many quantities have explicit representations
# and so we can double check if things work as expected.

using Random
using Empirikos
using Hypatia
Random.seed!(1)

n = 200

Zs = BinomialSample.(Int.(rand(Bernoulli(0.7), 200)), 1)
Zs_summary = summarize(Zs)

@test Zs_summary(BinomialSample(1,1)) == sum( ==(1), response.(Zs))
@test Zs_summary[BinomialSample(1,1)] == Zs_summary(BinomialSample(1,1))

@test length(Zs_summary) == 2
@test nobs(Zs_summary) == n

α = 0.11
floc_dkw = DvoretzkyKieferWolfowitz(;α=α)
floc_chisq = Empirikos.ChiSquaredFLocalization(;α=α)

pmf_at_1 = MarginalDensity(BinomialSample(1,1))

floc_dkw_interval = FLocalizationInterval(; flocalization = floc_dkw,
                                            solver = Hypatia.Optimizer,
                                            convexclass = DiscretePriorClass(0:0.001:1) )

floc_chisq_interval = FLocalizationInterval(; flocalization = floc_chisq,
                                            solver = Hypatia.Optimizer,
                                            convexclass = DiscretePriorClass(0:0.001:1) )


ci_at_1_dkw = confint(floc_dkw_interval, pmf_at_1, Zs_summary)

dkw_error = sqrt(log(2/α)/(2n))
hatf_1 =  Zs_summary(BinomialSample(1,1))/n

@test hatf_1 + dkw_error ≈ ci_at_1_dkw.upper
@test hatf_1 - dkw_error ≈ ci_at_1_dkw.lower

ci_at_1_chisq = confint(floc_chisq_interval, pmf_at_1, Zs_summary)

τsq = quantile(Chisq(1), 1-α)
chisq_f_center = (hatf_1 + τsq/(2n))/(1 + τsq/(n))
chisq_error = sqrt(τsq/n)/(1 + τsq/n)*sqrt( hatf_1*(1-hatf_1) +  τsq/(4n))

@test chisq_f_center +  chisq_error ≈ ci_at_1_chisq.upper
@test !(hatf_1 +  chisq_error ≈ ci_at_1_chisq.upper)
@test chisq_f_center -  chisq_error ≈ ci_at_1_chisq.lower

postmean_at_1 = PosteriorMean(BinomialSample(1,1))

ci_postmean_at_1_dkw = confint(floc_dkw_interval, postmean_at_1, Zs_summary)

@test ci_postmean_at_1_dkw.upper ≈ 1.0 atol = 1e-6
@test ci_postmean_at_1_dkw.lower ≈ hatf_1 - dkw_error atol = 1e-6

ci_postmean_at_1_chisq = confint(floc_chisq_interval, postmean_at_1, Zs_summary)

@test ci_postmean_at_1_chisq.upper ≈ 1.0 atol = 1e-6
@test ci_postmean_at_1_chisq.lower ≈ chisq_f_center -  chisq_error atol = 1e-6
@test !(ci_postmean_at_1_chisq.lower ≈ ci_postmean_at_1_dkw.lower)
