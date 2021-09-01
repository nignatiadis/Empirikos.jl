# Bernoulli is good sanity check since many quantities have explicit representations
# and so we can double check if things work as expected.

using Random
using Empirikos
using Hypatia
using Setfield

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







Random.seed!(1)

n = 200

Zs = BinomialSample.(Int.(rand(Bernoulli(0.7), 200)), 1)
Zs_summary = summarize(Zs)

comp_Zs = Empirikos.compound(Zs)
@test comp_Zs[1].probs == [1.0]
@test comp_Zs[1].vec == [BinomialSample(1)]
@test response(comp_Zs[1]) == response(Zs[1])

het_Zs = Empirikos.heteroskedastic(Zs)
@test het_Zs.probs == [1.0]
@test het_Zs.vec == [BinomialSample(1)]


amari_with_F = AMARI(;flocalization = (@set floc_dkw.α = 0.01),
                      solver = Hypatia.Optimizer,
                      convexclass = DiscretePriorClass(0:0.1:1),
                      modulus_model = Empirikos.ModulusModelWithF,
                      discretizer = Empirikos.integer_discretizer(0:1))

amari_without_F = AMARI(;flocalization = (@set floc_dkw.α = 0.01),
                      solver = Hypatia.Optimizer,
                      convexclass = DiscretePriorClass(0:0.1:1),
                      modulus_model = Empirikos.ModulusModelWithoutF,
                      discretizer = Empirikos.integer_discretizer(0:1))

for amari_ in (amari_with_F, amari_without_F)
    amari_fit_priormean = fit(amari_, Empirikos.PriorMean(), Zs)
    ci_priormean = confint(amari_fit_priormean, Empirikos.PriorMean(), Zs; level=1-α)
    @test confint(amari_, Empirikos.PriorMean(), Zs; level=1-α) == ci_priormean

    @test amari_fit_priormean.Q(BinomialSample(1,1)) ≈ 1.0 atol = 1e-6
    @test amari_fit_priormean.Q(BinomialSample(0,1)) ≈ 0.0 atol = 1e-6

    @test ci_priormean.estimate ≈ mean(response.(Zs)) atol = 1e-6
    @test ci_priormean.maxbias ≈ 0 atol = 1e-7
    @test amari_fit_priormean.max_bias ≈ 0 atol = 1e-7
    @test ci_priormean.se ≈ std(response.(Zs))/sqrt(nobs(Zs)) atol = 1e-7

    @test ci_priormean.lower ≈ ci_priormean.estimate - quantile(Normal(), 1-α/2)*ci_priormean.se
    @test ci_priormean.upper ≈ ci_priormean.estimate + quantile(Normal(), 1-α/2)*ci_priormean.se

    @test amari_fit_priormean.target === Empirikos.PriorMean()
    @test amari_fit_priormean.unit_var_proxy ≈ var(response.(Zs)) atol=0.01
    @test amari_fit_priormean.unit_var_proxy ≈ mean(response.(Zs))*(1- mean(response.(Zs))) atol=1e-5



    amari_fit_prior_second_mean = fit(amari_, Empirikos.PriorSecondMoment(), Zs)
    ci_second_mean = confint(amari_fit_prior_second_mean, Empirikos.PriorSecondMoment(), Zs; level=1-α)


    @test_throws ErrorException confint(amari_fit_prior_second_mean, Empirikos.PriorMean(), Zs; level=1-α)
    tmp_fit = fit((@set floc_dkw.α = 0.01), Zs)
    dkw_lb = mean(response, Zs) - tmp_fit.band

    @test amari_fit_prior_second_mean.max_bias ≈ dkw_lb*(1-dkw_lb)/2 rtol = 0.005
    @test amari_fit_prior_second_mean.Q(BinomialSample(1,1)) ≈ 1.0 - dkw_lb*(1-dkw_lb)/2  atol = 1e-3
    @test amari_fit_prior_second_mean.Q(BinomialSample(0,1)) ≈ - dkw_lb*(1-dkw_lb)/2  atol = 1e-3
    @test ci_second_mean.estimate ≈ mean(response.(Zs)) - dkw_lb*(1-dkw_lb)/2 atol = 1e-3
    @test ci_second_mean.lower < ci_second_mean.estimate - quantile(Normal(), 1-α/2)*ci_second_mean.se
    @test ci_second_mean.lower > ci_second_mean.estimate - quantile(Normal(), 1-α/2)*ci_second_mean.se - ci_second_mean.maxbias
    @test ci_second_mean.se ≈ std(response.(Zs))/sqrt(nobs(Zs)) atol = 1e-7
    @test amari_fit_prior_second_mean.unit_var_proxy ≈ mean(response.(Zs))*(1- mean(response.(Zs))) atol=1e-5

    both_cis = confint.(amari_, [Empirikos.PriorMean(), Empirikos.PriorSecondMoment()], Zs)
    @test getfield.(both_cis, :α) ≈ [0.05; 0.05]

    both_cis_kw = confint.(amari_, [Empirikos.PriorMean(), Empirikos.PriorSecondMoment()], Zs; level=0.95)
    @test both_cis_kw == both_cis

    both_cis_at_α = confint.(amari_, [Empirikos.PriorMean(), Empirikos.PriorSecondMoment()], Zs; level=1-α)
    @test both_cis_at_α == [ci_priormean; ci_second_mean]
end
