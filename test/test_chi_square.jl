using Empirikos
using Random
using MosekTools
using MultipleTesting
using HypothesisTests



1+1




σ²_prior = 4.0
ν_lik = 4
ν_prior = 3
n = 10_000
π1 = 0.05
v0 = 2


Z = Empirikos.ScaledChiSquareSample(2.0, ν_lik)
G = Empirikos.InverseScaledChiSquare(σ²_prior, ν_prior)


pvalue = Empirikos.limma_pvalue(2*1.96, Z, G)
_post = Empirikos.posterior(Z, G)


pvalue2_fun(β) = quadgk(u-> 2*ccdf(Normal(), β/sqrt(u))*pdf(_post, u), 0 , Inf)[1]
pvalue2_fun(2*1.96)

Zplus1 = Empirikos.ScaledChiSquareSample(2.0, ν_lik+1)
marginal = Empirikos.marginalize(Z, G)

marginal_plus1 = Empirikos.marginalize(Zplus1, G)
ssq = Zs.Z
int_lb = (abs2(2*1.96) + ν_lik * ssq)/(1+ν_lik)
pvalue3 = quadgk(t -> _const * t^(-(ν_lik-1)/2)*pdf(marginal_plus1, t) / (2*sqrt((ν_lik+1)t - ν_lik*ssq)), int_lb  , Inf)[1]

quadgk(x->pdf(marginal, x), 0, Inf)[1]
quadgk(x->pdf(marginal_plus1, x), 0, Inf)[1]

using SpecialFunctions
_const = ((1+1/ν_lik)^(-ν_lik/2)*sqrt(ν_lik +1)/ sqrt(pi)/pdf(marginal, ssq)) * ssq^(ν_lik/2-1) * gamma((ν_lik+1)/2) / gamma(ν_lik/2)

function pvalue3_fun(β)
    int_lb = (abs2(β) + ν_lik * ssq)/(1+ν_lik)
    quadgk(t -> _const * t^(-(ν_lik-1)/2)*pdf(marginal_plus1, t) / (sqrt((ν_lik+1)t - ν_lik*ssq)), int_lb  , Inf)[1]
end
pvalue3_fun(2*1.96)/pvalue2_fun(2*1.96)
pvalue3_fun(3*1.96)/pvalue2_fun(3*1.96)
pvalue3_fun(0.5)/pvalue2_fun(0.5)
pvalue3_fun(0.1)/pvalue2_fun(0.)


pvalue3/pvalue2
pvalue3_fun(0.5),pvalue2_fun(0.5)
pvalue3_fun(0.1),pvalue2_fun(0.1)


using QuadGK



loglikelihood(Zs, Empirikos.InverseScaledChiSquare(σ²_prior, ν_prior))




function single_sim()
    #σs_squared = rand(InverseGamma(Empirikos.InverseScaledChiSquare(σ²_prior, ν_prior)), n)
    σs_squared = fill(σ²_prior, n)
    β = fill(0.0, n)
    alt_idxs = Base.OneTo(floor(Int, π1*n))
    β[alt_idxs] .= 6.0
    #β[alt_idxs] = rand(Normal(), length(alt_idxs)) .* sqrt(v0) .* sqrt.(σs_squared[alt_idxs])
    Ss_squared = σs_squared ./ ν_lik .* rand(Chisq(ν_lik), n)
    Ss_squared = Empirikos.ScaledChiSquareSample.(Ss_squared, ν_lik)
    βs_hat = rand(Normal(), n) .* sqrt.(σs_squared) .+  β
    (βs_hat = βs_hat, Ss_squared = Ss_squared, βs = β, σs_squared = σs_squared)
end

Random.seed!(1)

function single_sim_eval()
    sim_res = single_sim()

    true_prior = Empirikos.InverseScaledChiSquare(σ²_prior, ν_prior)
    sigma_sq_grid = abs2.(0.0001:0.1:20)
    _prior = DiscretePriorClass(sigma_sq_grid)
    npmle_prior = fit(NPMLE(_prior, Mosek.Optimizer), sim_res.Ss_squared)

    npmle_pvalues = Empirikos.limma_pvalue.(sim_res.βs_hat, sim_res.Ss_squared, npmle_prior.prior)
    npmle_adjusted_pvalues = adjust(npmle_pvalues, BenjaminiHochberg())
    npmle_rjs = npmle_adjusted_pvalues .<= 0.1

    npmle_discoveries = max(sum( npmle_rjs), 1)
    npmle_true_discoveries = sum( npmle_rjs .&  (sim_res.βs .!= 0 ) )
    npmle_FDP = sum( npmle_rjs .*  (sim_res.βs .== 0 ) ) / npmle_discoveries

    oracle_pvalues = pvalue.(OneSampleZTest.( sim_res.βs_hat, sqrt(σ²_prior), 1))
    oracle_rjs = adjust(oracle_pvalues, BenjaminiHochberg()).<= 0.1
    oracle_discoveries = max(sum( oracle_rjs), 1)
    oracle_true_discoveries = sum( oracle_rjs .&  (sim_res.βs .!= 0 ) )
    oracle_FDP = sum( oracle_rjs .*  (sim_res.βs .== 0 ) ) / oracle_discoveries

    t_pvalues = pvalue.(OneSampleTTest.( sim_res.βs_hat,
    sqrt(ν_lik + 1) .* sqrt.(response.(sim_res.Ss_squared)), ν_lik + 1))
    t_rjs = adjust(t_pvalues, BenjaminiHochberg()).<= 0.1
    t_discoveries = max(sum( t_rjs), 1)
    t_true_discoveries = sum( t_rjs .&  (sim_res.βs .!= 0 ) )
    t_FDP = sum( t_rjs .*  (sim_res.βs .== 0 ) ) / t_discoveries

    return(
            t_true_discoveries = t_true_discoveries,
            t_discoveries = t_discoveries,
            t_FDP = t_FDP,
            oracle_true_discoveries = oracle_true_discoveries,
            oracle_discoveries = oracle_discoveries,
            oracle_FDP = oracle_FDP,
            npmle_true_discoveries = npmle_true_discoveries,
            npmle_discoveries = npmle_discoveries,
            npmle_FDP = npmle_FDP
    )
end

res = [ single_sim_eval() for i in 1:25]

npmle_FDR = mean(getfield.(res, :npmle_FDP))
#t_FDR = mean(getfield.(res, :t_FDP))
oracle_FDR = mean(getfield.(res, :oracle_FDP))

npmle_power = mean(getfield.(res, :npmle_true_discoveries))
#t_power = mean.(getfield.(res, :t_true_discoveries))
oracle_power = mean(getfield.(res, :oracle_true_discoveries))


histogram(tmp.t_pvalues[501:n], bins=0:0.02:1.0)
histogram(tmp.oracle_pvalues[501:n], bins=0:0.02:1.0)

histogram(tmp.npmle_pvalues[501:n], bins=0:0.02:1.0)

using Plots
using StatsPlots



hat_β = β .+ rand(Normal(), n) .* sqrt.(σs_squared)

Zs = Empirikos.ScaledChiSquareSample.(Zs, ν_lik)
using Optim




loglikelihood(Zs, Empirikos.InverseScaledChiSquare(σ²_prior, ν_prior))

ll_fun(v) = -loglikelihood(Zs, Empirikos.InverseScaledChiSquare(v[1], v[2]))
dfc = TwiceDifferentiableConstraints([0.0, 0.0], [Inf, Inf])


res = optimize(ll_fun, dfc, [1.0; 1.0], IPNewton())

estimated_prior = Empirikos.InverseScaledChiSquare(Optim.minimizer(res)...)


using StatsPlots
parametric_pvalues = Empirikos.limma_pvalue.(hat_β, Zs, estimated_prior)

true_pvalues = Empirikos.limma_pvalue.(hat_β, Zs, true_prior)

histogram(parametric_pvalues, bins=0:0.05:1.0)
histogram(true_pvalues, bins=0:0.05:1.0)


plot( -log10.(true_pvalues), -log10.(parametric_pvalues))

Empirikos.posterior(Zs[1], estimated_prior)

quantile(InverseGamma(true_prior), 0.99999)



using Hypatia


npmle_prior = fit(NPMLE(_prior, Hypatia.Optimizer), Zs)

maximum( support(npmle_prior.prior)[ probs(npmle_prior.prior) .> 1e-6])
maximum(σs_squared)


plot( support(npmle_prior.prior),  probs(npmle_prior.prior) , seriestype=:sticks, xlim=(0,100))
plot!( 0:0.01:100, estimated_prior )
loglikelihood(Zs, npmle_prior.prior)
loglikelihood(Zs, estimated_prior)



quantile(σs_squared)


using Plots

npmle_pvalues = Empirikos.limma_pvalue.(hat_β, Zs, npmle_prior.prior)
histogram(npmle_pvalues, bins=0:0.02:1.0)
histogram(oracle_pvalues, bins=0:0.02:1.0)
histogram(t_pvalues, bins=0:0.02:1.0)



plot( -log10.(true_pvalues), -log10.(npmle_pvalues), seriestype=:scatter, label="")
plot!(0:4, 0:4, linestyle=:dash, label="")
sum(npmle_pvalues .<= 0.05/n * 300)
sum(true_pvalues .<= 0.05/n *300)
