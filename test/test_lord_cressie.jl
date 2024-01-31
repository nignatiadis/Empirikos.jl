using Hypatia
using Empirikos
using Test


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
chisq_floc = Empirikos.ChiSquaredFLocalization(0.05)

floc_method_chisq = FLocalizationInterval(flocalization = chisq_floc,
                                       convexclass= gcal, solver=Hypatia.Optimizer)

chisq_cis = confint.(floc_method_chisq, postmean_targets, Zs)

lower_chisq_ci = getproperty.(chisq_cis, :lower)
upper_chisq_ci = getproperty.(chisq_cis, :upper)

@test lower_chisq_ci ≈ lower_test atol=1e-3
@test upper_chisq_ci ≈ upper_test atol=0.005



Zs_collapse = begin
	Zs_collapse = deepcopy(Zs)
	n0 = pop!(Zs_collapse.store, BinomialSample(0, 20))
	n1 = pop!(Zs_collapse.store, BinomialSample(1, 20))
	updated_keys =  [BinomialSample(Interval(0,1), 20); collect(keys(Zs_collapse))]
	updated_values = [n0+n1; collect(values(Zs_collapse))]
	Empirikos.summarize(updated_keys, updated_values)
end

discr = integer_discretizer(1:20)
Zs_collapse2 = discr(Zs)
discr(Zs_collapse) == Zs_collapse2

@test nobs(Zs_collapse) == nobs(Zs_collapse2)



floc_method_dkw = FLocalizationInterval(
							flocalization = DvoretzkyKieferWolfowitz(;α=0.05),
                            convexclass= gcal, solver=Hypatia.Optimizer)

fitted_dkw = fit(DvoretzkyKieferWolfowitz(;α=0.05), Zs_collapse)
fitted_dkw2 = fit(DvoretzkyKieferWolfowitz(;α=0.05), Zs_collapse2)

postmean_target = postmean_targets[2]
dkw_ci = confint(floc_method_dkw, postmean_target, Zs_collapse);
dkw_ci2 = confint(floc_method_dkw, postmean_target, Zs_collapse2);

@test dkw_ci == dkw_ci2

lam_chisq_withF = Empirikos.AMARI(
							convexclass=gcal,
                            flocalization = Empirikos.ChiSquaredFLocalization(0.01),
                            solver=Hypatia.Optimizer, discretizer=discr,
                            modulus_model = Empirikos.ModulusModelWithF
                           )

postmean_ci_lam_withF = confint(lam_chisq_withF, postmean_target, Zs_collapse2)

lam_chisq_withoutF = Empirikos.AMARI(
							convexclass=gcal,
                            flocalization = Empirikos.ChiSquaredFLocalization(0.01),
                            solver=Hypatia.Optimizer, discretizer=discr,
                            modulus_model = Empirikos.ModulusModelWithoutF
                           )

postmean_ci_lam_withoutF = confint(lam_chisq_withoutF, postmean_target, Zs_collapse2)

@test postmean_ci_lam_withoutF.lower ≈ postmean_ci_lam_withF.lower
@test postmean_ci_lam_withoutF.upper ≈ postmean_ci_lam_withF.upper
