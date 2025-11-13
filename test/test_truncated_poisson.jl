using StatsDiscretizations
using Hypatia

# Code below was shared by Mario Beraha and further modified 

Random.seed!(10)
n   = 2_000
G = Uniform(0.2, 6.0)
μs  = rand(G, n)            
raw = rand.(Poisson.(μs))                    
vals = filter(>(0), raw)                      
Zs  = Empirikos.TruncatedPoissonSample.(vals)
μmax    = max(8.0, maximum(vals))
grid = collect(exp.(range(log(1e-2), log(10), length=50)))      # a bit denser near 0
disc    = StatsDiscretizations.ExtendedFiniteGridDiscretizer(1:11)
gclass  = DiscretePriorClass(grid)


#grid_lo = collect(exp.(range(log(1e-2), log(6.0), length=200)))      # a bit denser near 0
#gclass  = DiscretePriorClass(grid_lo)     



Z_1 = sort(Zs, by = z->response(z))[1]
Z_max = sort(Zs, by = z->response(z))[end]
@test likelihood(Z_1, 3.0) ≈ likelihood(disc(Z_1), 3.0)

likelihood(Z_max, 10.0)



Zs = summarize(Zs)

floc = Empirikos.DvoretzkyKieferWolfowitz(α = 0.05)  # 95% band
method_floc = Empirikos.FLocalizationInterval(
    flocalization = floc,
    convexclass   = gclass,
    solver        = Hypatia.Optimizer,
)

method_amari = Empirikos.AMARI(
    convexclass   = gclass,
    flocalization = Empirikos.DvoretzkyKieferWolfowitz(α = 0.01), # pilot floc should be at lower α
    solver        = Hypatia.Optimizer,
    modulus_model=Empirikos.ModulusModelWithF,
    discretizer   = disc,
)

zs_target  = Empirikos.TruncatedPoissonSample.(1:4)
targets = Empirikos.PosteriorMean.(zs_target)
level = 0.95
cis_A = confint.(method_amari, targets, Zs; level=level)
cis_B = confint.(method_floc, targets, Zs; level=level)


# truncated_G 
tilted_G_norm = quadgk(μ-> pdf(G, μ)*ccdf(Poisson(μ), 0), 0.2, 6.0)[1]
pdf_G_tilt(μ) = pdf(G, μ)*ccdf(Poisson(μ), 0) / tilted_G_norm

zs_target_pois = PoissonSample.(1:4)

# true_targest, check tilting operation
true_targets_tilted = zeros(Float64, length(zs_target))
true_targets_tilted_equiv = zeros(Float64, length(zs_target))

for i in eachindex(zs_target)
    num = quadgk(μ-> pdf_G_tilt(μ)*μ*likelihood(zs_target[i], μ), 0.2, 6.0)[1]
    denom = quadgk(μ->  pdf_G_tilt(μ)*likelihood(zs_target[i], μ), 0.2, 6.0)[1]
    true_targets_tilted[i] = num / denom

    num = quadgk(μ-> pdf(G, μ)*μ*likelihood(zs_target_pois[i], μ), 0.2, 6.0)[1]
    denom = quadgk(μ->  pdf(G, μ)*likelihood(zs_target_pois[i], μ), 0.2, 6.0)[1]
    true_targets_tilted_equiv[i] = num / denom
end

true_targets_tilted
true_targets_tilted_equiv
@test true_targets_tilted ≈ true_targets_tilted_equiv

cis_A
cis_B

in_ci_A = [cis_A[i].lower <= true_targets_tilted[i] <= cis_A[i].upper for i in eachindex(cis_A)]
in_ci_B = [cis_B[i].lower <= true_targets_tilted[i] <= cis_B[i].upper for i in eachindex(cis_B)]

# In general only have probabilistic guarantee. Check just that at least one interval contains the true tilted value.
@test any(in_ci_A)
@test any(in_ci_B)

#tmp_grid = 0.01:0.01:6.0
#plot(tmp_grid, pdf.(G,tmp_grid))
#plot!(tmp_grid, pdf_G_tilt.(tmp_grid))

npmle = fit(Empirikos.NPMLE(convexclass = gclass, solver = Hypatia.Optimizer), Zs)
#plot(support(npmle.prior), probs(npmle.prior), seriestype=:sticks)

# More tests.

Zs = Butterfly.ebayes_samples()
target = PosteriorMean(TruncatedPoissonSample(4))

𝒢 = DiscretePriorClass(0.01:0.2:25)

floc_method_dkw = FLocalizationInterval(
							flocalization = DvoretzkyKieferWolfowitz(0.05),
                            convexclass = 𝒢, solver=Hypatia.Optimizer)


amari = AMARI(
            flocalization = DvoretzkyKieferWolfowitz(0.05),
                                convexclass = 𝒢, solver=Hypatia.Optimizer)


dkw_floc_cis = confint(floc_method_dkw, target, Zs)



amari = Empirikos.AMARI(convexclass = 𝒢,
                        flocalization = Empirikos.DvoretzkyKieferWolfowitz(0.01),
                        solver=Hypatia.Optimizer,
                        discretizer=StatsDiscretizations.ExtendedFiniteGridDiscretizer(1:20),
                        modulus_model=Empirikos.ModulusModelWithF
                        )


amari_cis = confint(amari, target, Zs)








# Test actual distribution
d = Empirikos.ZeroTruncatedPoisson(10.0)

@test sum(pdf.(d,1:100)) ≈ 1
@test pdf(d, 0) == 0

@test cdf(d,0) == 0
@test cdf(d,1) ≈ pdf(d, 1)
@test cdf(d,2) ≈ pdf(d, 1) + pdf(d,2)
@test cdf(d,100) ≈ 1
