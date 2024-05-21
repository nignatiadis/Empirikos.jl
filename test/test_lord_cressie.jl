using Hypatia
using Empirikos
using Test
using StatsDiscretizations
using Distributions
# We compare the chi^2 F-localization intervals against the intervals reported
# by Lord and Cressie.

lord_cressie = LordCressie.load_table()
Zs = LordCressie.ebayes_samples()

idx_test = [1, 13, 20]
z_cressie = lord_cressie.x[idx_test]
lower_test = lord_cressie.Lower1[idx_test]
upper_test = lord_cressie.Upper1[idx_test]

gcal = DiscretePriorClass(range(0.0,stop=1.0,length=300));

postmean_targets = Empirikos.PosteriorMean.(BinomialSample.(z_cressie,20));
chisq_floc = Empirikos.ChiSquaredFLocalization(α=0.05)

floc_method_chisq = FLocalizationInterval(flocalization = chisq_floc,
                                       convexclass= gcal, solver=Hypatia.Optimizer)

chisq_cis = confint.(floc_method_chisq, postmean_targets, Zs)

lower_chisq_ci = getproperty.(chisq_cis, :lower)
upper_chisq_ci = getproperty.(chisq_cis, :upper)

@test lower_chisq_ci ≈ lower_test atol=1e-3
@test upper_chisq_ci ≈ upper_test atol=0.005


discr = ExtendedFiniteGridDiscretizer(1:20)

Zs_discr = discr.(Zs)
Zs_collapse = summarize(collect(Zs_discr), collect(values(Zs)))

@test nobs(Zs_collapse)  == nobs(Zs)



floc_method_dkw = FLocalizationInterval(
							flocalization = DvoretzkyKieferWolfowitz(;α=0.05),
                            convexclass= gcal, solver=Hypatia.Optimizer)

fitted_dkw = fit(DvoretzkyKieferWolfowitz(;α=0.05), Zs_collapse)

postmean_target = postmean_targets[2]
dkw_ci = confint(floc_method_dkw, postmean_target, Zs_collapse)

lam_chisq_withF = Empirikos.AMARI(
							convexclass=gcal,
                            flocalization = Empirikos.ChiSquaredFLocalization(α=0.01, discretizer=discr),
                            solver=Hypatia.Optimizer, discretizer=discr,
                            modulus_model = Empirikos.ModulusModelWithF
                           )

postmean_ci_lam_withF = confint(lam_chisq_withF, postmean_target, Zs_collapse)

lam_chisq_withoutF = Empirikos.AMARI(
							convexclass=gcal,
                            flocalization = Empirikos.ChiSquaredFLocalization(α=0.01, discretizer=discr),
                            solver=Hypatia.Optimizer, discretizer=discr,
                            modulus_model = Empirikos.ModulusModelWithoutF
                           )

postmean_ci_lam_withoutF = confint(lam_chisq_withoutF, postmean_target, Zs_collapse)

@test postmean_ci_lam_withoutF.lower ≈ postmean_ci_lam_withF.lower
@test postmean_ci_lam_withoutF.upper ≈ postmean_ci_lam_withF.upper





