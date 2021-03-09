using Empirikos
using Test
using Hypatia
using Random
Zs_raw = NormalSample.(randn(10), rand(10))
Zs_compound = compound(Zs_raw)

Zs_standard = StandardNormalSample.(randn(100))
Zs_standard_compound = compound(Zs_standard)

@test Zs_standard_compound[1].vec == [StandardNormalSample()]

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

Random.seed!(1)
σs = rand(1:10,1000)
μs = randn(1000)
Zs = NormalSample.(μs, σs)
Zs_compound = compound(Zs)
@test length(Zs_compound[1].vec) == 10

Random.seed!(1)
Zs_binom_heterosk = BinomialSample.( sample(1:4, 100), sample(20:22,100))
Zs_binom_summary = summarize(Zs_binom_heterosk)
@test nobs(Zs_binom_summary) == 100
@test length(keys(Zs_binom_summary)) == 3*4
comp_binom = compound(Zs_binom_heterosk)
Zs_binom_comp_summary = summarize(comp_binom)
@test nobs(Zs_binom_comp_summary) == 100
@test length(keys(Zs_binom_comp_summary)) == 4

Zs_binom_summary_comp = compound(Zs_binom_summary)
@test nobs(Zs_binom_summary_comp) == 100
Empirikos.multiplicity(Zs_binom_comp_summary) == Empirikos.multiplicity(Zs_binom_summary_comp)
