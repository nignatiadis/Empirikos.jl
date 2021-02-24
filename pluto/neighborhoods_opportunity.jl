### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 1cae6098-7665-11eb-171c-39b12a001162
begin
	using Empirikos
	using DataFrames
	using MosekTools
	using LaTeXStrings
	using Plots
	using Gurobi
	pgfplotsx()
end

# ╔═╡ 33ff43d4-7665-11eb-38bd-b74e7d842b2e
md"""# Preliminaries

Load packages
"""

# ╔═╡ c20e1e20-7665-11eb-31d0-a9bbe2952d95
md"Load the Moving to Opportunities neighborhoods dataset (Chetty and Hendren, 2018) and remove missing entries"

# ╔═╡ 61ace78c-7665-11eb-14c0-0964539796fa
nbhood_csv = Empirikos.Neighborhoods.load_table() |> DataFrame |> dropmissing

# ╔═╡ d0a1ce7a-7665-11eb-12be-a3510bc83fc1
md"Wrap the data in a type that represents the Gaussian likelihood:"

# ╔═╡ 65596c02-7665-11eb-24a6-53194a8ef2ec
Zs = NormalSample.(nbhood_csv.p25_coef, nbhood_csv.p25_se)

# ╔═╡ 75a30c76-7665-11eb-1a42-5fb73c3fd435
md" Let us define the empirical Bayes targets we want to estimate."

# ╔═╡ 6a5dce64-7665-11eb-0640-074cb387b172
σs = [0.5 1.0 2.0]

# ╔═╡ 6d2de886-7665-11eb-38b2-918706170dab
targets = PosteriorMean.(NormalSample.(-3:0.1:3, σs))

# ╔═╡ 70a14292-7665-11eb-37bc-078e81fad4ce
target_names = [L"E[\mu \mid Z=z, \sigma=%$σ]" for σ in σs]

# ╔═╡ 86f6f028-7665-11eb-01df-b3f0fb425fc5
md"""# Smooth prior class

Throughout this Section we first assume that the true prior $G$ is smooth and lies in the class of mixtures of 

$$\left\{\mathcal{N}(\mu, 0.25^2), \mu \in \{-3,-2.9,\dotsc,3\} \right\}.$$
"""

# ╔═╡ 4738f08e-7666-11eb-1f8d-494095706ad9
gcal_smooth = MixturePriorClass(Normal.(-3:0.01:3, 0.25))

# ╔═╡ 4b7a72ee-7666-11eb-16cd-d9dcdfce85c2
md"We start by fitting the NPMLE to that prior class (this is not needed for the intervals)."

# ╔═╡ 5c57e074-7666-11eb-31f7-df6bc0db85e9
npmle_smooth_fit = fit(NPMLE(gcal_smooth, Mosek.Optimizer), Zs);

# ╔═╡ 67e70304-7666-11eb-3fac-a5b23821d5c4
plot(
    -3:0.001:3,
    x -> pdf(npmle_smooth_fit.prior, x),
    xguide = L"\mu",
    yguide = L"g(\mu)",
    title = "NPMLE prior estimate",
    label = nothing,
	size = (400,200)
)

# ╔═╡ 133d68e0-7667-11eb-02e6-5fd8f00ae23a
md" Start with confidence interval construction (compound DKW-F-Localization)"

# ╔═╡ 7641e23c-7666-11eb-33e6-554c4e2d3bc6
floc_method_smooth = FLocalizationInterval(;
    flocalization = DvoretzkyKieferWolfowitz(0.05),
    convexclass = gcal_smooth,
    solver = Gurobi.Optimizer,
)

# ╔═╡ ab5dd40a-7666-11eb-39da-81b36a92c705
postmean_cis_smooth = confint.(floc_method_smooth, targets, compound(Zs))


# ╔═╡ 7ddc4a5e-7667-11eb-0990-135e43d168b3
md" Also evaluate the plugin estimates based on the NPMLE."

# ╔═╡ afff51f8-7666-11eb-326c-a1d8ec9d6bb0
targets_smooth_npmle = targets.(npmle_smooth_fit.prior);

# ╔═╡ b6500f2a-7666-11eb-018b-33a641ed3650
begin
plots_smooth = plot.(eachcol(postmean_cis_smooth), xlabel = L"z", label = "")
plot!.(
    plots_smooth,
    Ref([-3.0; 3.0]),
    Ref([-3.0, 3.0]),
    label = "",
    seriestype = :line,
    linestyle = :dot,
    seriescolor = :gray,
)
plot!.(
    plots_smooth,
    Ref(-3:0.1:3.0),
    eachcol(targets_smooth_npmle),
    label = "",
    seriestype = :line,
    linecolor = :purple,
    linestyle = :dash,
)
plot(plots_smooth..., ylabel = target_names, layout=(1,3), size=(600,200))
end

# ╔═╡ 8850d39c-7667-11eb-27b1-73efcdae0f3e
md"""# Discrete class
Now instead we make a more standard nonparametric EB assumption, i.e. that $G$ is an arbitrary discrete distribution support on the grid $-3,-2.9,\dotsc,2.9,3$ and repeat the same steps as we did for the smooth class above.
"""

# ╔═╡ ab9429c6-7667-11eb-3c9b-e188eddd8702
discrete_class = Empirikos.DiscretePriorClass(-3:0.01:3)

# ╔═╡ bfd3b20c-7668-11eb-26df-15c438553ebe
npmle_discrete_fit = fit(NPMLE(discrete_class, Mosek.Optimizer), Zs);

# ╔═╡ f57e856e-7669-11eb-2b81-5906651d88dc
plot(
    support(npmle_discrete_fit.prior),
    probs(npmle_discrete_fit.prior),
    seriestype = :sticks,
    xguide = L"\mu",
    yguide = L"g(\mu)",
    title = "NPMLE prior estimate",
    label = nothing,
	size = (400,200)
)

# ╔═╡ 2eeedd3c-766a-11eb-18c2-eb97bade511e
floc_method_discrete = FLocalizationInterval(;
    flocalization = DvoretzkyKieferWolfowitz(0.05),
    convexclass = discrete_class,
    solver = Gurobi.Optimizer,
)

# ╔═╡ bce4edc8-7667-11eb-0d6c-5d7b99fead6c


# ╔═╡ c2316fb8-7667-11eb-00a9-1bef1ce72652
postmean_cis_discrete = confint.(floc_method_discrete, targets, compound(Zs))

# ╔═╡ c52a446a-7667-11eb-344b-4124556ca32a
targets_discrete_npmle = targets.(npmle_discrete_fit.prior)

# ╔═╡ ca99b43a-7667-11eb-04f9-ff24c8341d77
begin
plots_discrete = plot.(eachcol(postmean_cis_discrete), xlabel = L"z", label = "")
plot!.(
    plots_discrete,
    Ref([-3.0; 3.0]),
    Ref([-3.0, 3.0]),
    label = "",
    seriestype = :line,
    linestyle = :dot,
    seriescolor = :gray,
)
plot!.(
    plots_discrete,
    Ref(-3:0.1:3.0),
    eachcol(targets_discrete_npmle),
    label = "",
    seriestype = :line,
    linecolor = :purple,
    linestyle = :dash,
)
plot(plots_discrete..., ylabel = target_names, layout=(1,3), size=(600,200))
end

# ╔═╡ Cell order:
# ╟─33ff43d4-7665-11eb-38bd-b74e7d842b2e
# ╠═1cae6098-7665-11eb-171c-39b12a001162
# ╟─c20e1e20-7665-11eb-31d0-a9bbe2952d95
# ╠═61ace78c-7665-11eb-14c0-0964539796fa
# ╟─d0a1ce7a-7665-11eb-12be-a3510bc83fc1
# ╠═65596c02-7665-11eb-24a6-53194a8ef2ec
# ╟─75a30c76-7665-11eb-1a42-5fb73c3fd435
# ╠═6a5dce64-7665-11eb-0640-074cb387b172
# ╠═6d2de886-7665-11eb-38b2-918706170dab
# ╠═70a14292-7665-11eb-37bc-078e81fad4ce
# ╟─86f6f028-7665-11eb-01df-b3f0fb425fc5
# ╠═4738f08e-7666-11eb-1f8d-494095706ad9
# ╟─4b7a72ee-7666-11eb-16cd-d9dcdfce85c2
# ╠═5c57e074-7666-11eb-31f7-df6bc0db85e9
# ╠═67e70304-7666-11eb-3fac-a5b23821d5c4
# ╟─133d68e0-7667-11eb-02e6-5fd8f00ae23a
# ╠═7641e23c-7666-11eb-33e6-554c4e2d3bc6
# ╠═ab5dd40a-7666-11eb-39da-81b36a92c705
# ╟─7ddc4a5e-7667-11eb-0990-135e43d168b3
# ╠═afff51f8-7666-11eb-326c-a1d8ec9d6bb0
# ╠═b6500f2a-7666-11eb-018b-33a641ed3650
# ╟─8850d39c-7667-11eb-27b1-73efcdae0f3e
# ╠═ab9429c6-7667-11eb-3c9b-e188eddd8702
# ╠═bfd3b20c-7668-11eb-26df-15c438553ebe
# ╠═f57e856e-7669-11eb-2b81-5906651d88dc
# ╠═2eeedd3c-766a-11eb-18c2-eb97bade511e
# ╟─bce4edc8-7667-11eb-0d6c-5d7b99fead6c
# ╠═c2316fb8-7667-11eb-00a9-1bef1ce72652
# ╠═c52a446a-7667-11eb-344b-4124556ca32a
# ╠═ca99b43a-7667-11eb-04f9-ff24c8341d77
