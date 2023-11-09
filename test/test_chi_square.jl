using Empirikos
using Random
using MosekTools
using MultipleTesting
using HypothesisTests

using Optim
using QuadGK
using LinearAlgebra
using Expectations

using Cubature
Gmix = DiscreteNonParametric([1.0;0.5], [1/2; 1/2])
Gsingle = DiscreteNonParametric([1.0], [1.0])


pval(z, G) = dot(probs(G), 2*ccdf.( Normal.(0, support(G)), abs(z)))

pval(1.96, DiscreteNonParametric([1.0], [1.0]))

quadgk(z-> (pval(z, Gsingle) <= 0.01)*pdf(Normal(),z), -7, +7)

quadgk(z-> (pval(z, Gmix) <= 0.01)*pdf(Normal(),z), -7, +7)


function limma_type_pval_inner(params; t=0.01, ν=3, G=Gsingle)
    z = params[1]
    s_squared = params[2]
    S_squared = Empirikos.ScaledChiSquareSample(s_squared, ν)
    Gmix = Empirikos.posterior(S_squared, G)
    pdf(Normal(),z)*(pval(z, Gmix) <= t)
end
using QuadGK

function limma_outer_pval(params; ν=3, G=Gsingle)
    z = params[1]
    s_squared = params[2]
    S_squared = Empirikos.ScaledChiSquareSample(s_squared, ν)
    lik = likelihood_distribution(S_squared, 1)
    inner_pval = limma_type_pval_inner(params; ν=ν, G=G)
    pdf(Normal(), z)

end



function limma_level(G, ν, t)
    E = expectation(
    likelihood_distribution(Empirikos.ScaledChiSquareSample(1.0, ν),1)
    )
    E(s_sq -> quadgk(z->limma_type_pval_inner([z;s_sq]; G=G, ν=ν, t=t), -7, 7)[1])
end



νs = Base.OneTo(60)
_level = limma_level.(Ref(Gmix), νs, 0.01)



using Plots
pgfplotsx()

theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    grid = nothing,
    frame = :box,
    legendfonthalign = :left,
    thickness_scaling = 1.1
)

plot(νs, _level, seriestype=:scatter, label="",  xguide="ν")
plot!([1; 60], [0.01; 0.01], linestyle=:dash,
    label="",
    ylabel="Average significance", xlim=(1,60),
    xticks=1:6:60, ylim=(0,0.0205))
savefig("avg_significance_violation.pdf")




using GR
gr()
using StatsPlots
histogram2d(u,v)
using GRUtils

u = randn(100_000)
v = randn(100_000)
x = cos.(2pi .* (0:0.001:1))
y = sin.(2pi .* (0:0.001:1))


plot(x,y, seriestype=:scatter, showaxis=false)
hold(true)
shade(u,v, colormap=-8,showaxis=false)
GRUtils.plot(x, y)
GRUtils.grid(false)
GRUtils.xticks(false)
GRUtils.yticks(false)

GRUtils.savefig("hello.pdf")

limma_level(Gmix, 50, 0.01)
using LaTeXStrings
using Expectations
Exp
quadgk( s_sq ->     )



function limma_type_pval(params; ν=3, G=Gsingle)
    z = params[1]
    s_squared = params[2]
    S_squared = Empirikos.ScaledChiSquareSample(s_squared, ν)
    lik = likelihood_distribution(S_squared, 1)
    Gmix = Empirikos.posterior(S_squared, G)
    pdf(Normal(),z)*pdf(lik, s_squared)*(pval(z, Gmix) <= 0.01)
end

using Cubature

mean(Empirikos.posterior(Empirikos.ScaledChiSquareSample(1, 3), Gmix))
limma_type_pval([1.96;1.0];G=Gsingle)



hcubature(z->limma_type_pval(z;G=Gmix), [-7; 1e-5], [+7;20])





using CSV

methylation = CSV.File("methylation.csv")

methylation = CSV.File("methylation_promoter.csv")


limma_prior_var = 0.0498492 #promoter
limma_ν_prior = 3.659067 #promoter
using LaTeXStrings
pgfplotsx()

theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    grid = nothing,
    frame = :box,
    legendfonthalign = :left,
    thickness_scaling = 1.4,
    size = (420, 330),
)

histogram_plot = histogram(abs2.(methylation.sigma_hat), fill="lightgrey",
    normalize=true, fillalpha=0.5, linewidth=0.3,
    bins=400, xlim=(0,1), ylim=(0,10.5), label="Histogram",
    xlabel=L"S_i^2", ylabel="Density", legend=:topright)
#limma_prior_var = 0.0492029
#limma_ν_prior = 3.957179

limma_prior = Empirikos.InverseScaledChiSquare(limma_prior_var, limma_ν_prior)

methylation_Zs = Empirikos.ScaledChiSquareSample.(abs2.(methylation.sigma_hat), 4)
mu_hat = methylation.mu_hat
limma_pvalues =   Empirikos.limma_pvalue.(mu_hat, methylation_Zs, limma_prior)


ttest_pvalues = 2*ccdf.( TDist(4), abs.(mu_hat)./methylation.sigma_hat  )

minimum(ttest_pvalues)


limma_pvalues ≈ methylation.limma_pvalue


extrema(response.(methylation_Zs))
sqrt.(extrema(response.(methylation_Zs)))

sigma_sq_grid = abs2.(0.0077:0.001:2.2)
#sigma_sq_grid = abs2.(0.05:0.001:2.2)

_prior = DiscretePriorClass(sigma_sq_grid)

#_prior = MixturePriorClass(Empirikos.InverseScaledChiSquare.(sigma_sq_grid, 20))

disc = interval_discretizer(sigma_sq_grid)
Zs_summary = summarize(disc.(methylation_Zs))

npmle_prior = fit(NPMLE(_prior, Mosek.Optimizer), Zs_summary)

loglikelihood(methylation_Zs, npmle_prior.prior)
loglikelihood(methylation_Zs, limma_prior)

loglikelihood(methylation_Zs, Dirac(mean(abs2.(methylation.sigma_hat))))


npmle_pvalues =   Empirikos.limma_pvalue.(mu_hat, methylation_Zs, npmle_prior.prior)

minimum(npmle_pvalues)
minimum(limma_pvalues)

sum(adjust(ttest_pvalues, BenjaminiHochberg(), ) .<= 0.1)

sum(adjust(limma_pvalues, BenjaminiHochberg(), ) .<= 0.1)
sum(adjust(npmle_pvalues, BenjaminiHochberg(), ) .<= 0.1)

sum(adjust(limma_pvalues[filter_idx], BenjaminiHochberg(), ) .<= 0.05)
sum(adjust(npmle_pvalues[filter_idx], BenjaminiHochberg(), ) .<= 0.05)

histogram(npmle_pvalues)
histogram(limma_pvalues)
histogram(ttest_pvalues)


using Plots


plot(support(npmle_prior.prior), 1000 * probs(npmle_prior.prior), seriestype=:line,
     xlim=(0, 0.7), markersize = 2, markerstrokewidth = 0,
     legend = :topright, label="NPMLE",
     color = :blue,
     xlabel=L"\sigma_i^2", ylabel="Density")
#limma_f = u->pdf(limma_prior,u^2 )*2*u
plot!(u->pdf(limma_prior,u ), label="Inverse Gamma", color=:darkorange)
savefig("methylation_estimated_priors.pdf")


savefig(histogram_plot, "methylation_variance_histogram.pdf")

var_grid_refine =  1e-5:0.001:1
var_grid_refine_samples = Empirikos.ScaledChiSquareSample.(var_grid_refine, 4)

marginal_pdf_limma= pdf.(limma_prior, var_grid_refine_samples)
marginal_pdf_npmle = pdf.(npmle_prior.prior, var_grid_refine_samples)

histogram_plot = histogram(abs2.(methylation.sigma_hat), fill="lightgrey",
    normalize=true, fillalpha=0.5, linewidth=0.3,
    bins=500, xlim=(0,0.7), ylim=(0,13), label="Histogram",
    xlabel=L"S_i^2", ylabel="Density", legend=:topright)

savefig(histogram_plot, "methylation_variance_histogram.pdf")


plot!(histogram_plot, var_grid_refine,
    [marginal_pdf_npmle marginal_pdf_limma], label=["NPMLE" "Inverse Gamma"],
    color = [:blue :darkorange],
    linewidth=0.7)

savefig(histogram_plot, "methylation_variance_histogram_and_pdfs.pdf")

using StatsBase
ecdf_limma_ps = ecdf(limma_pvalues)
ecdf_npmle_ps = ecdf(npmle_pvalues)
ps = 0.00:0.01:1.0

plot(ps, [ecdf_npmle_ps.(ps) ecdf_limma_ps.(ps)],
    linestyle=[:dot :solid], linewidth=[1.3 1.1],
    linealpha=0.7, color=[:blue :darkorange],
    label=["NPMLE" "Inverse Gamma"], legend=:bottomright,
    xlabel=L"P_i", ylabel="Distribution")

plot(ps, ecdf_npmle_ps.(ps),
    color=:blue, markershape=:circle,
    label="NPMLE", legend=:bottomright, markersize=0.3,
    xlabel=L"P_i", ylabel="Distribution")

savefig("methylation_cdfs.pdf")

plot(1:10,1:10)

filter_idx =  methylation.sigma_hat .< 1
histogram(abs2.(methylation.sigma_hat[filter_idx]), normalize=true, fill="white",
    bins=100)







mean(limma_prior)
std(limma_prior)
mean(npmle_prior.prior)
median(InverseGamma(limma_prior))

median(npmle_prior.prior)
#std(npmle_prior.prior)



plot(support(npmle_prior.prior), 1000 * probs(npmle_prior.prior), seriestype=:scatter,
     xlim=(0, 1), markersize = 2, markerstrokewidth = 0, markershape=:x)

plot(support(npmle_prior.prior), 1000 * probs(npmle_prior.prior), seriestype=:line,
     xlim=(0, 1), markersize = 2, markerstrokewidth = 0)
plot!(u->pdf(limma_prior,u ))

plot(sigma_sq_grid , u->pdf(npmle_prior.prior,u ))


plot(sqrt.(support(npmle_prior.prior)), 50 * probs(npmle_prior.prior), seriestype=:sticks,
     xlim=(0, 1))
#limma_f ./ sum(limma_f, sqrt.(support(npmle_prior.prior)))

var_grid = 1e-5:0.005:7
var_grid_samples = Empirikos.ScaledChiSquareSample.(var_grid, 4)
postmeans = PosteriorMean.(var_grid_samples)
npmle_postmeans = postmeans.(npmle_prior.prior)
limma_postmeans = postmeans.(limma_prior)

plot(var_grid, [npmle_postmeans limma_postmeans], label=["npmle" "limma"])

pvals_npmle_fun = Empirikos.limma_pvalue.(1.0, var_grid_samples, npmle_prior.prior)
pvals_limma_fun = Empirikos.limma_pvalue.(1.0, var_grid_samples, limma_prior)


var_grid = 0.001:0.001:0.5
var_grid_samples = Empirikos.ScaledChiSquareSample.(var_grid, 4)

pvals_npmle_fun = Empirikos.limma_pvalue.(1.0 .* sqrt.(var_grid), var_grid_samples, npmle_prior.prior)
pvals_limma_fun = Empirikos.limma_pvalue.(1.0 .* sqrt.(var_grid), var_grid_samples, limma_prior)

pvals_npmle_fun2 = Empirikos.limma_pvalue.(2.0 .* sqrt.(var_grid), var_grid_samples, npmle_prior.prior)
pvals_limma_fun2 = Empirikos.limma_pvalue.(2.0 .* sqrt.(var_grid), var_grid_samples, limma_prior)
pvals_limma_fun3 = Empirikos.limma_pvalue.(3.0 .* sqrt.(var_grid), var_grid_samples, limma_prior)
pvals_npmle_fun3 = Empirikos.limma_pvalue.(3.0 .* sqrt.(var_grid), var_grid_samples, npmle_prior.prior)

plot(var_grid, [pvals_npmle_fun pvals_limma_fun], label=["npmle 1" "limma 1"])
plot(var_grid, [pvals_npmle_fun2 pvals_limma_fun2], label=["npmle 2" "limma 2"])
plot!(var_grid, )
plot(var_grid, [pvals_npmle_fun3 pvals_limma_fun3], label=["npmle 3" "limma 3"])


effects = 0.0:0.1:3.0

pvals_npmle_var005 = Empirikos.limma_pvalue.(effects.* sqrt.(0.3), Empirikos.ScaledChiSquareSample(0.3, 4), npmle_prior.prior)
pvals_limma_var005 = Empirikos.limma_pvalue.(effects.* sqrt.(0.3), Empirikos.ScaledChiSquareSample(0.3, 4), limma_prior)

plot(effects, [pvals_npmle_var005 pvals_limma_var005], label=["npmle 1" "limma 1"])



var_grid_refine =  1e-5:0.001:1
var_grid_refine_samples = Empirikos.ScaledChiSquareSample.(var_grid_refine, 4)

marginal_pdf_limma= pdf.(limma_prior, var_grid_refine_samples)
marginal_pdf_npmle = pdf.(npmle_prior.prior, var_grid_refine_samples)


filter_idx =  methylation.sigma_hat .< 1
histogram(abs2.(methylation.sigma_hat[filter_idx]), normalize=true, fill="white",
    bins=100)

plot!(var_grid_refine, [marginal_pdf_npmle marginal_pdf_limma], label=["npmle" "limma"],
linesize=3)

plot(1:10,1:10)
plot(limma_pvalues, npmle_pvalues, markersize=1,
    markerstrokewidth = 0, seriestype=:scatter, label="",
    markeralpha=0.005, color=[:blue :darkorange])
plot!([0,1], [0,1], linestyle=:dash, color=:black, label="",
    xlabel=L"P_i \textrm{ (Inverse Gamma)}", ylabel=L"P_i \textrm{ (NPMLE)}")
savefig("methylation_pvalue_vs_pvalue.pdf")
using StatsPlots



histogram(npmle_pvalues, fill=:blue,
    normalize=true, fillalpha=0.2, linewidth=0.3,
    bins=50, xlim=(0,1), ylim=(0,2.2), label="NPMLE",
    xlabel=L"P_i", ylabel="Density", legend=:topright)
histogram!(limma_pvalues, fill=:darkorange,
    normalize=true, fillalpha=0.2, linewidth=0.3,
    bins=50, xlim=(0,1), ylim=(0,2.2), label="Inverse Gamma",
    xlabel=L"P_i", ylabel="Density", legend=:topright)

savefig("methylation_pvalue_histograms.pdf")


findmin(npmle_pvalues)
findmin(limma_pvalues)

methylation[202561]

cor( limma_pvalues, npmle_pvalues)
marginalhist(limma_pvalues, npmle_pvalues)
plot!([0;1]; [0,1])

plot(-log10.(limma_pvalues), -log10.(npmle_pvalues), markersize=1,
    markerstrokewidth = 0, seriestype=:scatter,
    aspect_ratio=:equal)
plot!([-8;8]; [-8,8])
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

loglikelihood(Zs, Empirikos.InverseScaledChiSquare(σ²_prior, ν_prior))



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
