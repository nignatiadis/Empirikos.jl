using LinearAlgebra
using Hypatia
using Random
using Empirikos
using Test
using JuMP
G = Normal(1,0.3)
F_G = marginalize(StandardNormalSample(), G)
Random.seed!(1)

Zs = StandardNormalSample.(rand(F_G, 1000))

gcal = MixturePriorClass(Normal.(-3:0.05:3, 0.2))

ghat = StatsBase.fit( NPMLE(;convexclass=gcal, solver=Hypatia.Optimizer),  Zs)



floc = StatsBase.fit(DvoretzkyKieferWolfowitz(0.01), Zs)


discr = Empirikos.Discretizer(-3:0.01:3)

amari_withF  = AMARI(;convexclass=gcal, flocalization=floc,solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = ghat, modulus_model=Empirikos.ModulusModelWithF)

amari_withoutF  = AMARI(;convexclass=gcal, flocalization=floc,solver=Hypatia.Optimizer, discretizer=discr,
              plugin_G = ghat, modulus_model=Empirikos.ModulusModelWithoutF)

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


_init_with_F = Empirikos.initialize_method(amari_withF, target, Zs)
Empirikos.set_δ!(_init_with_F.modulus_model,  0.01);

_init_without_F = Empirikos.initialize_method(amari_withoutF, target, Zs)
Empirikos.set_δ!(_init_without_F.modulus_model,  0.01);


@test JuMP.dual(_init_with_F.modulus_model.bound_delta) ≈ JuMP.dual(_init_without_F.modulus_model.bound_delta) rtol=0.0001
