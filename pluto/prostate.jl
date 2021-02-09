### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ 1fbdb6c6-4bf4-11eb-3aeb-9548bde2b1a5
begin
	using Empirikos
	using Plots
	using PGFPlotsX
	using LaTeXStrings
	using MosekTools
	using JuMP
	using Setfield
end

# ╔═╡ 3cf2bafa-4bf4-11eb-3482-ff09ca17af71
md"""
# Identifying genes associated with prostate cancer

## Load packages and dataset
"""

# ╔═╡ 473d8a28-4bf4-11eb-0151-8b3f3d286eda
begin
	pgfplotsx()
	deleteat!(PGFPlotsX.CUSTOM_PREAMBLE,
			  Base.OneTo(length(PGFPlotsX.CUSTOM_PREAMBLE)))
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\PP}[2][]{\mathbb{P}_{#1}\left[#2\right]}")
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}")
end;

# ╔═╡ 17be2c0c-4bf5-11eb-25ca-cfe4e5fb5485
theme(:default;
      background_color_legend = :transparent,
      foreground_color_legend = :transparent,
      grid=nothing,
      frame=:box,
      thickness_scaling=1.3)

# ╔═╡ acfc4fc0-4bf4-11eb-32dc-dfe1652ac006
Zs = Prostate.ebayes_samples();

# ╔═╡ f4bd73b8-4bf4-11eb-3678-655227b0895b
md"""
## Marginal distribution of z-scores

### DKW-F-Localization
"""

# ╔═╡ 0731218a-4bf5-11eb-10f4-81241477b265
dkw_floc = DvoretzkyKieferWolfowitz(0.05)

# ╔═╡ 0d36dbee-4bf5-11eb-3036-7bcf68579a46
fitted_dkw = fit(dkw_floc, Zs)

# ╔═╡ 246264dc-4bf5-11eb-2f9d-b5c7d137a9b3
dkw_plot = plot(fitted_dkw, subsample=300, label="DKW band",
     xlab=L"z", ylab=L"\widehat{F}_n(z)",  size=(380,280))

# ╔═╡ 4dd1f99a-4bf5-11eb-3691-9db09ab324df
savefig(dkw_plot, "prostate_dkw_band.tikz")

# ╔═╡ 362effcc-4bf5-11eb-1d84-3133de3f4825
md"### KDE-F-Localization"

# ╔═╡ 5ad6b662-4bf5-11eb-24fb-d141933957ca
infty_floc = Empirikos.InfinityNormDensityBand(a_min=-3.0,a_max=3.0);

# ╔═╡ 605a545e-4bf5-11eb-3020-6d9c53e39a8e
fitted_infty_floc = fit(infty_floc, Zs)

# ╔═╡ 69e87366-4bf5-11eb-3fbb-8f7bf325d366
prostate_kde_plot = begin
	prostate_marginal_plot = histogram(response.(Zs), bins=50, normalize=true,
          label="Histogram", fillalpha=0.4, linealpha=0.4, fillcolor=:lightgray,
          size=(380,280), xlims=(-4.5,4.5))
	plot!(prostate_marginal_plot, fitted_infty_floc,
       	  label="KDE band", xlims=(-4.5,4.5),
          yguide=L"\widehat{f}_n(z)", xguide=L"z")
	plot!([-3.0;3.0], seriestype=:vline,
		  linestyle=:dot, label=nothing, color=:lightgrey)
end

# ╔═╡ 750e2af0-4bf6-11eb-3374-1f224647d4e5
savefig("prostate_kde_band.tikz")

# ╔═╡ aa99cc74-4bf6-11eb-2e3f-81de213db00d
 md"## Confidence intervals"

# ╔═╡ df1513be-4bf6-11eb-1b1c-fd650fd994fe
quiet_mosek = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)

# ╔═╡ c4bc3468-4bf4-11eb-1710-f7638d026b91
gcal_locmix = MixturePriorClass(Normal.(-3:0.05:3, 0.25));

# ╔═╡ bd7a5362-4bf4-11eb-1f2c-79ff3e915f89
gcal_scalemix = Empirikos.set_defaults(GaussianScaleMixtureClass(), Zs; hints = Dict(:grid_scaling => 1.1))


# ╔═╡ 9b6b8f8a-4bf6-11eb-002d-7fa04ddde0d3
discr = Empirikos.Discretizer(-3.0:0.005:3.0)

# ╔═╡ ce945cca-4bf6-11eb-29bc-4b2514ae395d
floc_method_dkw_locmix = FLocalizationInterval(flocalization = dkw_floc,
                               convexclass = gcal_locmix, solver = quiet_mosek)


# ╔═╡ e9e81b9c-4bf6-11eb-1c3c-d56076f6858b
floc_method_kde_locmix = FLocalizationInterval(flocalization = infty_floc,
                               convexclass = gcal_locmix, solver = quiet_mosek)


# ╔═╡ 04c72ec6-4bf7-11eb-3265-5f69502b9f0b

lam_kde_locmix = Empirikos.AMARI(
						convexclass = gcal_locmix,
						flocalization = (@set infty_floc.α=0.01),
                        discretizer=discr,
                        solver=quiet_mosek, )



# ╔═╡ 31bd322c-4bf7-11eb-007c-d7583730cd8e
floc_method_dkw_scalemix = FLocalizationInterval(flocalization = dkw_floc,
                               convexclass = gcal_scalemix, solver = quiet_mosek)


# ╔═╡ 3d5b459c-4bf7-11eb-3825-6b75cc4099f3
floc_method_kde_scalemix = FLocalizationInterval(flocalization = infty_floc,
                               convexclass = gcal_scalemix, solver = quiet_mosek)


# ╔═╡ 474bdc10-4bf7-11eb-0099-cdf3150001a2
lam_kde_scalemix = Empirikos.AMARI(
						convexclass = gcal_scalemix,
						flocalization = (@set infty_floc.α=0.01),
                        discretizer=discr,
                        solver=quiet_mosek, )

# ╔═╡ 5fe96288-4bf7-11eb-1805-1f7f7815e035
ts= -3:0.2:3

# ╔═╡ 65d3b2de-4bf7-11eb-3ab5-a13a5b5556ef
postmean_targets = Empirikos.PosteriorMean.(StandardNormalSample.(ts))

# ╔═╡ 6ade8f74-4bf7-11eb-2b46-51436f6bf297
lfsrs = Empirikos.PosteriorProbability.(StandardNormalSample.(ts),
										Interval(0,nothing))


# ╔═╡ 7c0fd708-4bf7-11eb-12e0-45ecc7f9af7a
postmean_ci_dkw_locmix = confint.(floc_method_dkw_locmix, postmean_targets, Zs)

# ╔═╡ 85c63cc4-4bf7-11eb-2688-2b63ea65b2e7
postmean_ci_kde_locmix = confint.(floc_method_kde_locmix, postmean_targets, Zs)

# ╔═╡ 89a047c2-4bf7-11eb-3c79-15fe2fcdcb1c
postmean_ci_lam_locmix = confint.(lam_kde_locmix, postmean_targets, Zs)

# ╔═╡ a262fe94-4bf7-11eb-1153-6f76128cbcd0
postmean_locmix_plot = begin
	postmean_locmix_plot = plot(ts, postmean_ci_kde_locmix, label="KDE-F-Loc",
		fillcolor=:darkorange, fillalpha=0.5, ylim=(-2.55,2.55))
	plot!(postmean_locmix_plot, ts, postmean_ci_dkw_locmix, label="DKW-F-Loc",
		show_ribbon=false, alpha=0.9, color=:black)
	plot!(postmean_locmix_plot, ts, postmean_ci_lam_locmix, label="Amari",
		show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
	plot!(postmean_locmix_plot, [-3.0;3.0], [-3.0; 3.0], seriestype=:line,
		linestyle=:dot, label=nothing, color=:lightgrey)
	plot!(postmean_locmix_plot, xlabel = L"z",
		ylabel=L"\EE{\mu \mid Z=z}", size=(380,280))
end


# ╔═╡ 622669fa-4bf8-11eb-08fd-a3bd67230729
savefig(postmean_locmix_plot, "prostate_locmix_postmean.tikz")

# ╔═╡ 6930b1ae-4bf8-11eb-23c7-e5be40b7a283
postmean_ci_dkw_scalemix = confint.(floc_method_dkw_scalemix, postmean_targets, Zs)

# ╔═╡ 6fba4f94-4bf8-11eb-2177-87f38a9b73d2
postmean_ci_kde_scalemix = confint.(floc_method_kde_scalemix, postmean_targets, Zs)

# ╔═╡ 7c308bb4-4bf8-11eb-249a-0f6d933a530c
postmean_ci_lam_scalemix = confint.(lam_kde_scalemix, postmean_targets, Zs)

# ╔═╡ 8abdac16-4bf8-11eb-3b0d-a73d7b28bb7c
postmean_scalemix_plot = begin
	postmean_scalemix_plot = plot(ts, postmean_ci_kde_scalemix,
		label="KDE-F-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(-2.55,2.55))
	plot!(postmean_scalemix_plot, ts, postmean_ci_dkw_scalemix,
		label="DKW-F-Loc",show_ribbon=false, alpha=0.9, color=:black)
	plot!(postmean_scalemix_plot, ts, postmean_ci_lam_scalemix,
		label="Amari",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
	plot!(postmean_scalemix_plot, [-3.0;3.0], [-3.0; 3.0], seriestype=:line,
		linestyle=:dot, label=nothing, color=:lightgrey)
	plot!(postmean_scalemix_plot, xlabel = L"z",
		ylabel=L"\EE{\mu \mid Z=z}", size=(380,280))
end

# ╔═╡ eb851e8a-4bf8-11eb-353a-e5c394ae5294
savefig(postmean_scalemix_plot, "prostate_scalemix_postmean.tikz")

# ╔═╡ 1e0db04c-4bf9-11eb-1bde-9dfa4f81f279
lfsr_ci_dkw_locmix = confint.(floc_method_dkw_locmix, lfsrs, Zs)

# ╔═╡ 2700eeda-4bf9-11eb-27ba-c5c6fe33980d
lfsr_ci_kde_locmix = confint.(floc_method_kde_locmix, lfsrs, Zs)

# ╔═╡ 2eba35aa-4bf9-11eb-1bf0-0be32d4bff85
lfsr_ci_lam_locmix = confint.(lam_kde_locmix, lfsrs, Zs)

# ╔═╡ a30dce02-4bfa-11eb-1197-5b69f20af457
lfsr_locmix_plot = begin
	lfsr_locmix_plot = plot([-3;3], [0.5; 0.5], seriestype=:line,
		linestyle=:dot, label=nothing, color=:lightgrey)
	plot!(lfsr_locmix_plot, ts, lfsr_ci_kde_locmix,
		label="KDE-F-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(0,1))
	plot!(lfsr_locmix_plot, ts, lfsr_ci_dkw_locmix,
		label="DKW-F-Loc",show_ribbon=false, alpha=0.9, color=:black)
	plot!(lfsr_locmix_plot, ts, lfsr_ci_lam_locmix,
		label="Amari",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
	plot!(lfsr_locmix_plot, xlabel = L"z",
		ylabel=L"\PP{\mu \geq 0 \mid Z=z}", size=(380,280))
end

# ╔═╡ 716b8220-4bf9-11eb-1a28-33dbf08e138c
savefig(lfsr_locmix_plot, "prostate_locmix_lfsr.tikz")

# ╔═╡ 9afa4b4a-4bf9-11eb-2a16-531b86e6b9a9
lfsr_ci_dkw_scalemix = confint.(floc_method_dkw_scalemix, lfsrs, Zs)

# ╔═╡ 9e06f2b8-4bf9-11eb-1bad-b5a1d26339ee
lfsr_ci_kde_scalemix = confint.(floc_method_kde_scalemix, lfsrs, Zs)

# ╔═╡ a1ba1e12-4bf9-11eb-00c9-496602f4eb5c
lfsr_ci_lam_scalemix = confint.(lam_kde_scalemix, lfsrs, Zs)

# ╔═╡ f2414706-4bfa-11eb-1654-9bdcc7418db8
lfsr_scalemix_plot = begin
	lfsr_scalemix_plot = plot([-3;3], [0.5; 0.5], seriestype=:line,
		linestyle=:dot, label=nothing, color=:lightgrey)
	plot!(lfsr_scalemix_plot, ts, lfsr_ci_kde_scalemix,
		label="KDE-F-Loc", fillcolor=:darkorange, fillalpha=0.5, ylim=(0,1))
	plot!(lfsr_scalemix_plot, ts, lfsr_ci_dkw_scalemix,
		label="DKW-F-Loc",show_ribbon=false, alpha=0.9, color=:black)
	plot!(lfsr_scalemix_plot, ts, lfsr_ci_lam_scalemix,
		label="Amari",show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
	plot!(lfsr_scalemix_plot, xlabel = L"z",
		ylabel=L"\PP{\mu \geq 0 \mid Z=z}", size=(380,280))
end


# ╔═╡ f7a59c42-4bfa-11eb-27df-cf611213fefc
savefig(lfsr_scalemix_plot, "prostate_scalemix_lfsr.tikz")


# ╔═╡ Cell order:
# ╟─3cf2bafa-4bf4-11eb-3482-ff09ca17af71
# ╠═1fbdb6c6-4bf4-11eb-3aeb-9548bde2b1a5
# ╠═473d8a28-4bf4-11eb-0151-8b3f3d286eda
# ╠═17be2c0c-4bf5-11eb-25ca-cfe4e5fb5485
# ╠═acfc4fc0-4bf4-11eb-32dc-dfe1652ac006
# ╟─f4bd73b8-4bf4-11eb-3678-655227b0895b
# ╠═0731218a-4bf5-11eb-10f4-81241477b265
# ╠═0d36dbee-4bf5-11eb-3036-7bcf68579a46
# ╠═246264dc-4bf5-11eb-2f9d-b5c7d137a9b3
# ╠═4dd1f99a-4bf5-11eb-3691-9db09ab324df
# ╟─362effcc-4bf5-11eb-1d84-3133de3f4825
# ╠═5ad6b662-4bf5-11eb-24fb-d141933957ca
# ╠═605a545e-4bf5-11eb-3020-6d9c53e39a8e
# ╠═69e87366-4bf5-11eb-3fbb-8f7bf325d366
# ╠═750e2af0-4bf6-11eb-3374-1f224647d4e5
# ╟─aa99cc74-4bf6-11eb-2e3f-81de213db00d
# ╠═df1513be-4bf6-11eb-1b1c-fd650fd994fe
# ╠═c4bc3468-4bf4-11eb-1710-f7638d026b91
# ╠═bd7a5362-4bf4-11eb-1f2c-79ff3e915f89
# ╠═9b6b8f8a-4bf6-11eb-002d-7fa04ddde0d3
# ╠═ce945cca-4bf6-11eb-29bc-4b2514ae395d
# ╠═e9e81b9c-4bf6-11eb-1c3c-d56076f6858b
# ╠═04c72ec6-4bf7-11eb-3265-5f69502b9f0b
# ╠═31bd322c-4bf7-11eb-007c-d7583730cd8e
# ╠═3d5b459c-4bf7-11eb-3825-6b75cc4099f3
# ╠═474bdc10-4bf7-11eb-0099-cdf3150001a2
# ╠═5fe96288-4bf7-11eb-1805-1f7f7815e035
# ╠═65d3b2de-4bf7-11eb-3ab5-a13a5b5556ef
# ╠═6ade8f74-4bf7-11eb-2b46-51436f6bf297
# ╠═7c0fd708-4bf7-11eb-12e0-45ecc7f9af7a
# ╠═85c63cc4-4bf7-11eb-2688-2b63ea65b2e7
# ╠═89a047c2-4bf7-11eb-3c79-15fe2fcdcb1c
# ╠═a262fe94-4bf7-11eb-1153-6f76128cbcd0
# ╠═622669fa-4bf8-11eb-08fd-a3bd67230729
# ╠═6930b1ae-4bf8-11eb-23c7-e5be40b7a283
# ╠═6fba4f94-4bf8-11eb-2177-87f38a9b73d2
# ╠═7c308bb4-4bf8-11eb-249a-0f6d933a530c
# ╠═8abdac16-4bf8-11eb-3b0d-a73d7b28bb7c
# ╠═eb851e8a-4bf8-11eb-353a-e5c394ae5294
# ╠═1e0db04c-4bf9-11eb-1bde-9dfa4f81f279
# ╠═2700eeda-4bf9-11eb-27ba-c5c6fe33980d
# ╠═2eba35aa-4bf9-11eb-1bf0-0be32d4bff85
# ╠═a30dce02-4bfa-11eb-1197-5b69f20af457
# ╠═716b8220-4bf9-11eb-1a28-33dbf08e138c
# ╠═9afa4b4a-4bf9-11eb-2a16-531b86e6b9a9
# ╠═9e06f2b8-4bf9-11eb-1bad-b5a1d26339ee
# ╠═a1ba1e12-4bf9-11eb-00c9-496602f4eb5c
# ╠═f2414706-4bfa-11eb-1654-9bdcc7418db8
# ╠═f7a59c42-4bfa-11eb-27df-cf611213fefc
