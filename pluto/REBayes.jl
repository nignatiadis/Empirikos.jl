### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ ebbb8758-126a-11eb-3682-31a80e9f8437
begin 
	using Empirikos
	using JuMP
	using RCall
	using LaTeXStrings
	using Plots
	using StatsPlots
	using MosekTools
	using Random
end

# ╔═╡ ff3c85b6-126a-11eb-3fc0-635d27bf1d7e
md"""
# Reproducing the REBayes R vignette


The goal of this notebook is showcase the functionality of the `Empirikos.jl` package by (partially) reproducing results shown in the [REBayes vignette](https://cran.r-project.org/web/packages/REBayes/vignettes/rebayes.pdf).
"""

# ╔═╡ 88f13064-12ba-11eb-2329-b1b7196a3af4
pgfplotsx()

# ╔═╡ 675ba12a-12b8-11eb-0791-77781a1619fa
md"""
## Gaussian Mixture models

### Needles and haystacks
"""

# ╔═╡ 8a5b07c4-12b8-11eb-0884-19f987050ba4
normal_Zs = StandardNormalSample.(rand(MersenneTwister(100), Normal(), 1000) .+ 										  [fill(0,900); fill(2,100)]);

# ╔═╡ cf8c5152-12b8-11eb-220e-b9750bf9cc9f
normal_npmle = fit(NPMLE(DiscretePriorClass(), Mosek.Optimizer), normal_Zs);

# ╔═╡ e7ab7a1e-12b9-11eb-2dc6-0180b9df937d
normal_postmean_targets =  PosteriorMean.(sort(normal_Zs));

# ╔═╡ 1731df1a-12b9-11eb-2aeb-2762823871f4
begin
normal_pl_marginal = plot(marginalize(NormalSample(1.0), 
		                  DiscreteNonParametric([0.0;2.0], [0.9;0.1])), 									  components=false, label=nothing, xguide="z",
                          yguide = "f(z)", 
						  title = "True mixture")
normal_pl_npmle_prior = plot(support(normal_npmle.prior),
	 	  					 probs(normal_npmle.prior), seriestype=:sticks,
	                         xguide = L"\mu", yguide=L"g(\mu)",
		                     title = "NPMLE prior estimate", label=nothing)
normal_pl_posterior_mean = plot( response.(location.(normal_postmean_targets)),
		                         normal_postmean_targets.(normal_npmle.prior),
	                             label=nothing, title="NPMLE posterior mean",
	                             xguide="z", yguide=L"E[\mu \mid Z=z]")
plot(normal_pl_marginal, normal_pl_npmle_prior, normal_pl_posterior_mean,
	 size=(750,250), layout=(1,3))
end

# ╔═╡ fa7eae00-126a-11eb-1fe6-09bb4826877e
md"""
## Binomial NPMLE

We start by loading the Tacks data by Beckett and Diaconis.
"""

# ╔═╡ 470e7048-126b-11eb-3eb1-7bca6628f555
R"library(REBayes)";

# ╔═╡ ed1ab854-126b-11eb-399f-677c45bf7ecd
tacks_tbl = rcopy(R"tacks");

# ╔═╡ 04590064-126c-11eb-2023-9767e8bead48
first(tacks_tbl,3)

# ╔═╡ 5b6c41cc-126c-11eb-315d-8752bfde9d9a
md"""
Let us convert the table as a vector of Binomial samples
"""

# ╔═╡ 89cd6128-126b-11eb-2164-9d634900847a
tack_Zs = BinomialSample.(tacks_tbl.x, Int64.(tacks_tbl.k))

# ╔═╡ c774a158-126b-11eb-2241-a9b6cf80c681
tacks_npmle = fit(NPMLE(DiscretePriorClass(), Mosek.Optimizer), tack_Zs);

# ╔═╡ 345669c6-126e-11eb-2b12-a976e63e3335
plot(support(tacks_npmle.prior),
	 probs(tacks_npmle.prior),
	 seriestype=:sticks,
	 xguide = "p", 
	 yguide = "g(p)", 
	 size=(350,250),
	 label=nothing)

# ╔═╡ 5a8ad83e-126e-11eb-2d85-6581c5379a4b
md"""
# Poisson NPMLE
"""

# ╔═╡ b43cdce2-126e-11eb-0da0-fd2ec7d7bbad
norberg = rcopy(R"Norberg"); first(norberg,3)

# ╔═╡ cccfbc70-126e-11eb-0439-4552285a634d
norberg_Zs = Empirikos.PoissonSample.(norberg.Death, norberg.Exposure ./ 344);

# ╔═╡ f6e28740-1273-11eb-1464-ab04ba549059
extrema(response.(norberg_Zs) ./ nuisance_parameter.(norberg_Zs))

# ╔═╡ 3460571e-126f-11eb-176b-a3445db55987
sum(response.(norberg_Zs) .== 0) # how many zeros in the samples?

# ╔═╡ 08c5dc30-1276-11eb-01c1-4baf4dece1d5
mosek_lowreltol = optimizer_with_attributes(Mosek.Optimizer, 
					"MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 10^(-8))

# ╔═╡ e7e4d35a-126f-11eb-34aa-dbb75f86c6a6
norberg_npmle = fit(NPMLE(DiscretePriorClass(), 
		                  mosek_lowreltol;
		                  prior_grid_length=1000), 
					norberg_Zs);

# ╔═╡ 80fcb26e-12a3-11eb-0977-3f37bdd5759e
norberg_gamma_mle = fit(ParametricMLE(Gamma()), norberg_Zs)

# ╔═╡ 56671378-1271-11eb-1a8f-3f289512a146
begin
pl = plot(plot(support(norberg_npmle.prior),
	 	  probs(norberg_npmle.prior)./sum(probs(norberg_npmle.prior)),
	      seriestype=:sticks,
	      xguide = L"\theta", 
	      yguide = L"g(\theta)", 
		  legend=:topright, 
		  label="NPMLE"),
	 plot(support(norberg_npmle.prior),
	 	  (probs(norberg_npmle.prior) ./ sum(probs(norberg_npmle.prior))).^(1/3),
	      seriestype=:sticks,
	      xguide = L"\theta", 
	      yguide = L"\sqrt[3]{g(\theta)}",
		  legend=:topright,
		  label="NPMLE"),
	size=(700,250))
plot!(pl[1], support(norberg_npmle.prior), 
		     x->pdf(norberg_gamma_mle, x), color=:purple, label="Γ MLE")
end

# ╔═╡ db81cb56-12a4-11eb-3b5b-910d159781fd
norgberg_postmean_gamma = (PosteriorMean.(norberg_Zs)).(norberg_gamma_mle)

# ╔═╡ 316b4056-12a5-11eb-129f-bf07f3962a39
norgberg_postmean_npmle = (PosteriorMean.(norberg_Zs)).(norberg_npmle.prior)

# ╔═╡ a0f0629e-12a5-11eb-3d7e-c92f58bb13ad
begin
	plot(norgberg_postmean_gamma, norgberg_postmean_npmle, seriestype=:scatter,
		 xguide= "Γ MLE posterior mean", 
		 yguide = "NPMLE posterior mean", label=nothing,
		 size=(300,250), aspect_ratio=1)
	plot!(norgberg_postmean_gamma, norgberg_postmean_gamma,
		 linestyle=:dash, label=nothing)
end

# ╔═╡ Cell order:
# ╟─ff3c85b6-126a-11eb-3fc0-635d27bf1d7e
# ╠═ebbb8758-126a-11eb-3682-31a80e9f8437
# ╠═88f13064-12ba-11eb-2329-b1b7196a3af4
# ╟─675ba12a-12b8-11eb-0791-77781a1619fa
# ╠═8a5b07c4-12b8-11eb-0884-19f987050ba4
# ╠═cf8c5152-12b8-11eb-220e-b9750bf9cc9f
# ╠═e7ab7a1e-12b9-11eb-2dc6-0180b9df937d
# ╠═1731df1a-12b9-11eb-2aeb-2762823871f4
# ╟─fa7eae00-126a-11eb-1fe6-09bb4826877e
# ╠═470e7048-126b-11eb-3eb1-7bca6628f555
# ╠═ed1ab854-126b-11eb-399f-677c45bf7ecd
# ╠═04590064-126c-11eb-2023-9767e8bead48
# ╟─5b6c41cc-126c-11eb-315d-8752bfde9d9a
# ╠═89cd6128-126b-11eb-2164-9d634900847a
# ╠═c774a158-126b-11eb-2241-a9b6cf80c681
# ╠═345669c6-126e-11eb-2b12-a976e63e3335
# ╟─5a8ad83e-126e-11eb-2d85-6581c5379a4b
# ╠═b43cdce2-126e-11eb-0da0-fd2ec7d7bbad
# ╠═cccfbc70-126e-11eb-0439-4552285a634d
# ╠═f6e28740-1273-11eb-1464-ab04ba549059
# ╠═3460571e-126f-11eb-176b-a3445db55987
# ╠═08c5dc30-1276-11eb-01c1-4baf4dece1d5
# ╠═e7e4d35a-126f-11eb-34aa-dbb75f86c6a6
# ╠═80fcb26e-12a3-11eb-0977-3f37bdd5759e
# ╠═56671378-1271-11eb-1a8f-3f289512a146
# ╠═db81cb56-12a4-11eb-3b5b-910d159781fd
# ╠═316b4056-12a5-11eb-129f-bf07f3962a39
# ╠═a0f0629e-12a5-11eb-3d7e-c92f58bb13ad
