using Empirikos
using Test
using Hypatia


tbl = Empirikos.CollinsLangman.load_table()

# test against last row in Van Houwelingen (1993), Table I
@test sum(tbl.NT) == 1215
@test sum(tbl.XT) == 254
@test sum(tbl.NC) == 1192
@test sum(tbl.XC) == 272


Zs_all =  Empirikos.CollinsLangman.ebayes_samples()

ors =  [((Z.Z + 0.5)/(Z.m1 - Z.Z +0.5)) / ((Z.n - Z.Z + 0.5)/(Z.m2 - Z.n + Z.Z + 0.5)) for Z ∈ Zs_all]
@test round.(ors; digits=2) == tbl.OR




Zs = Zs_all[getfield.(Zs_all, :n) .!= 0]
θs = -4:0.01:4
likelihoods = [likelihood(z, θ) for θ ∈ θs, z ∈ Zs]

scaled_likelihoods = likelihoods ./ sum(likelihoods; dims= 1)


#using Plots
#plot(θs, scaled_likelihoods)

# Compare to values of likelihood reported in van Houwelingen (1993)

two_point_prior = DiscreteNonParametric([-0.03; 1.10], [0.79; 0.21])
@test loglikelihood(Zs, two_point_prior) ≈ -51.76 atol = 0.005


# Do we recover the NPMLE reported in Van Houwelingen?
npmle = NPMLE(;convexclass = DiscretePriorClass(-4:0.01:4),
               solver = Hypatia.Optimizer)

npmle_fit = fit(npmle, Zs)

#plot(support(npmle_fit.prior), probs(npmle_fit.prior), series=:sticks)

@test loglikelihood(Zs, npmle_fit.prior) ≈ -51.76 atol = 0.005
