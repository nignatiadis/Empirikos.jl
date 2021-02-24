Zs_raw = NormalSample.(randn(10), rand(10))
Zs_compound = compound(Zs_raw)

Zs_standard = StandardNormalSample.(randn(100))
Zs_standard_compound = compound(Zs_standard)

@test cdf.(Normal(0,2), Zs_standard) == cdf.(Normal(0,2), Zs_standard)
@test pdf.(Normal(0,2), Zs_standard) == pdf.(Normal(0,2), Zs_standard)


dkw_loc_standard_compound = fit(DvoretzkyKieferWolfowitz(0.05*exp(1)), Zs_standard_compound)
dkw_loc_standard = fit(DvoretzkyKieferWolfowitz(0.05), Zs_standard)

@test dkw_loc_standard.homoskedastic
@test !dkw_loc_standard_compound.homoskedastic

@test dkw_loc_standard.band ≈ dkw_loc_standard_compound.band

target = PosteriorMean(NormalSample(2.0, 2.0))

gcal = DiscretePriorClass(-2:0.2:2)
floc_method_standard = FLocalizationInterval(flocalization=DvoretzkyKieferWolfowitz(0.05),
                             convexclass=gcal,
                             solver=Hypatia.Optimizer)
floc_method_compound = FLocalizationInterval(flocalization=DvoretzkyKieferWolfowitz(0.05*exp(1)),
                             convexclass=gcal,
                             solver=Hypatia.Optimizer)


ci1 = confint(floc_method_standard, target, Zs_standard)
ci2 = confint(floc_method_compound, target, Zs_standard_compound)

@test ci1.lower ≈ ci2.lower
@test ci1.upper ≈ ci2.upper
@test ci1.α ≈ 0.05
@test ci2.α ≈ 0.05*exp(1)
