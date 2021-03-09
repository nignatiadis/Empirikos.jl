using Hypatia
using Random
using Empirikos
using LinearAlgebra
using Test
using JuMP
G = Normal(1,0.3)
F_G = marginalize(StandardNormalSample(), G)
Random.seed!(1)

Zs = StandardNormalSample.(rand(F_G, 1000))

gcal = MixturePriorClass(Normal.(-3:0.5:3, 0.2))

ghat = StatsBase.fit( NPMLE(;convexclass=gcal, solver=Hypatia.Optimizer),  Zs)



floc = StatsBase.fit(DvoretzkyKieferWolfowitz(;α=0.01), Zs)


discr = Empirikos.Discretizer(-3:0.2:3)

amari_withF  = AMARI(;convexclass=gcal, flocalization=floc,solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = ghat.prior, modulus_model=Empirikos.ModulusModelWithF)

amari_withoutF  = AMARI(;convexclass=gcal, flocalization=floc,solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = ghat.prior, modulus_model=Empirikos.ModulusModelWithoutF)

target = MarginalDensity(NormalSample(1.0, 0.5))
target(G)


fit_withoutF = StatsBase.fit(amari_withoutF, target, Zs)
fit_withF = StatsBase.fit(amari_withF, target, Zs)


@test fit_withoutF.Q(StandardNormalSample(2.0)) ≈ fit_withF.Q(StandardNormalSample(2.0)) atol = 0.001

ci_withF = confint(amari_withF, target, Zs)
ci_withoutF = confint(amari_withoutF, target, Zs)

@test ci_withF.lower ≈ ci_withoutF.lower rtol =0.001
@test ci_withF.upper ≈ ci_withoutF.upper rtol =0.001
@test ci_withF.halflength ≈ ci_withoutF.halflength rtol =0.001
@test ci_withF.estimate ≈ ci_withoutF.estimate rtol =0.001
@test ci_withF.maxbias ≈ ci_withoutF.maxbias rtol =0.001


_init_with_F = Empirikos.initialize_method(amari_withF, target, Zs)
Empirikos.set_δ!(_init_with_F.modulus_model,  0.01);

_init_without_F = Empirikos.initialize_method(amari_withoutF, target, Zs)
Empirikos.set_δ!(_init_without_F.modulus_model,  0.01);


@test JuMP.dual(_init_with_F.modulus_model.bound_delta) ≈ JuMP.dual(_init_without_F.modulus_model.bound_delta) rtol=0.0001



amari_withF  = AMARI(;convexclass=gcal, flocalization=floc,solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = ghat.prior, modulus_model=Empirikos.ModulusModelWithF)

amari_withoutF  = AMARI(;convexclass=gcal, flocalization=floc,solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = ghat.prior, modulus_model=Empirikos.ModulusModelWithoutF)



Random.seed!(1)
G = Normal(0, 1.0)
σs = rand(1:10,1000)./5
zs = rand(G, 1000) .+ randn(1000) .* σs
Zs = NormalSample.(μs, σs)
@test length(Empirikos.heteroskedastic(Zs).vec) == 10

gcal_mix = MixturePriorClass(Normal.(-4:0.2:4, 0.5))

discr = Empirikos.Discretizer(-4:0.1:4)

amari  = AMARI(;convexclass=gcal, flocalization=DvoretzkyKieferWolfowitz(;α=0.01),
              solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = NPMLE(;convexclass=gcal, solver=Hypatia.Optimizer))


targets = [MarginalDensity( NormalSample(1.0, 0.3)), Empirikos.PriorDensity(  1.5)]
for target in targets
    amari_fit = StatsBase.fit(amari, target, Zs)
    amari_ci = confint(amari_fit, target, Zs)
    lp_biases = Empirikos.worst_case_bias_lp(amari_fit.method, amari_fit.Q, target)
    @show target
    @show lp_biases
    @test lp_biases.maxbias ≈ amari_ci.maxbias atol = 0.00001
    @test lp_biases.maxbias ≈ -lp_biases.minbias atol = 0.00001
end



postmean_target = PosteriorMean( NormalSample(1.0, 0.5))
@test_throws String confint(amari_fit, postmean_target, Zs)

postmean_ci = confint(amari, postmean_target, Zs)





postmean_ci_01 = confint(amari, postmean_target, Zs; α=0.1)
@test postmean_ci_01.upper - postmean_ci_01.lower < postmean_ci.upper - postmean_ci.lower


#Qs = amari_fit.Q.(Zs)

#X = [fill(1, length(Qs)) Qs]
#X_hats = diag(X*inv(X'X)*X')
#scatter( std.(Zs), X_hats)
