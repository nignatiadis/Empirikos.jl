### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 33cbdf58-4bd1-11eb-2c37-81d5f476828e
begin
	using DataFrames
	using Empirikos
	using Plots
	using PGFPlotsX
	using LaTeXStrings
	using MosekTools
	using JuMP
end

# ╔═╡ 7cd14ada-4bd1-11eb-137b-0591983ae782
md"""
# Predicting performance in Psychometric Tests

## Load packages and dataset
"""

# ╔═╡ b852c54c-4bd2-11eb-27c9-6fbfa3dc09e7
begin
	pgfplotsx()
	deleteat!(PGFPlotsX.CUSTOM_PREAMBLE,
			  Base.OneTo(length(PGFPlotsX.CUSTOM_PREAMBLE)))
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\PP}[2][]{\mathbb{P}_{#1}\left[#2\right]}")
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}")
end;

# ╔═╡ bd3d470e-4bd1-11eb-2b6a-3342f8a8a272
lord_cressie = LordCressie.load_table() |> DataFrame

# ╔═╡ 1e6e4c8a-4bd2-11eb-29ae-9bcb1d13a026
Zs = Empirikos.MultinomialSummary(BinomialSample.(lord_cressie.x, 20),
                                  lord_cressie.N1);


# ╔═╡ 52bbfa58-4bd4-11eb-0f3b-1977a2671524
md"""
## Plot empirical frequencies (Fig. 1a)
"""

# ╔═╡ 7fcce91e-4bd2-11eb-3ba0-893699bbb1d6
empirical_probs = StatsBase.weights(Zs)/nobs(Zs)

# ╔═╡ a197f656-4bd2-11eb-1d34-15acb373e216
lord_cressie_freq = plot(
	 0:20,
	 sqrt.(empirical_probs), seriestype=:sticks, frame=:box,
     grid=nothing, color=:grey, markershape=:circle,
     markerstrokealpha = 0, ylim=(-0.001,sqrt(0.15)),
     xguide=L"z",yguide=L"\sqrt{\hat{f}_n(z)}",thickness_scaling=1.3,
     label=nothing, size=(500,350)
	)

# ╔═╡ 39c4faae-4bd4-11eb-2de6-0fb52b666531
# savefig(lord_cressie_freq, "lord_cressie_pdf.tikz")

# ╔═╡ 7e1c518e-4bd4-11eb-0b09-75616ed63a2e
md"""
## Construct confidence intervals

We first create a discretization object that will combine all counts ≤ 1.
"""

# ╔═╡ d3f6af54-8161-11eb-0dd9-41ae82c23455
discr = integer_discretizer(1:20) 

# ╔═╡ caaf9fce-4bd4-11eb-3c4e-318bd5bc838b
Zs_collapse = discr(Zs)

# ╔═╡ 955f125c-4bd2-11eb-2e6d-195f45c0fb6d
quiet_mosek = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)


# ╔═╡ 98836b02-4bd4-11eb-3f00-f118ee2aa731
𝒢 = DiscretePriorClass(range(0.0,stop=1.0,length=300));

# ╔═╡ 9dd42a9c-4bd4-11eb-0c4a-f59835f20ed0
postmean_targets = Empirikos.PosteriorMean.(BinomialSample.(0:20,20));

# ╔═╡ db6d0e50-4bd4-11eb-2ba1-5349d9a0ad62
md""" ### $\chi^2-F$-localization intervals"""

# ╔═╡ a51a7982-4bd4-11eb-09b5-434543d65565
chisq_floc = Empirikos.ChiSquaredFLocalization(0.05)

# ╔═╡ aa690a8e-4bd4-11eb-3f7c-c36955f21c08
floc_method_chisq = FLocalizationInterval(flocalization = chisq_floc,
                                       convexclass= 𝒢, solver=quiet_mosek)

# ╔═╡ c24e0e60-4bd4-11eb-048c-0b02d946fc95
chisq_cis = confint.(floc_method_chisq, postmean_targets, Zs_collapse)

# ╔═╡ 14790a00-4bd5-11eb-395e-d37b63309906
lower_chisq_ci = getproperty.(chisq_cis, :lower)

# ╔═╡ 25dbbcf2-4bd5-11eb-10a9-09afccc8faf7
upper_chisq_ci = getproperty.(chisq_cis, :upper)

# ╔═╡ 2e7825ee-4bd5-11eb-0433-a30a514b316a
begin
	lord_cressie.lower_chisq = reverse(lower_chisq_ci)
	lord_cressie.upper_chisq = reverse(upper_chisq_ci)
	lord_cressie
end

# ╔═╡ e64a9d2e-4bd4-11eb-10bc-4902fa5a6aed
md""" ### DKW $F$-localization intervals"""

# ╔═╡ ecca61e8-4bd4-11eb-1f1d-5f7a677e157c
floc_method_dkw = FLocalizationInterval(
							flocalization = DvoretzkyKieferWolfowitz(0.05),
                            convexclass = 𝒢, solver=quiet_mosek)


# ╔═╡ 0316c28e-4bd5-11eb-00c2-2150f66f1721
dkw_cis = confint.(floc_method_dkw, postmean_targets, Zs_collapse);


# ╔═╡ b7f27c2a-4bd5-11eb-3b2b-63532c39571b
lower_dkw_ci = getproperty.(dkw_cis, :lower)

# ╔═╡ be60c940-4bd5-11eb-0972-377395236cff
upper_dkw_ci = getproperty.(dkw_cis, :upper)

# ╔═╡ c26c3fb0-4bd5-11eb-3607-05043b4df440
md"""### Amari intervals"""

# ╔═╡ dec4fae4-4bd5-11eb-337d-0152aba1d793
lam_chisq = Empirikos.AMARI(
							convexclass = 𝒢,
                            flocalization = Empirikos.ChiSquaredFLocalization(0.01),
                            solver=quiet_mosek,
                            discretizer=discr
                            )


# ╔═╡ 64050d52-4bd6-11eb-2fa6-8bf4fbee5ba3
postmean_ci_lam = confint.(lam_chisq, postmean_targets, Zs_collapse)

# ╔═╡ 6d844faa-4bd6-11eb-236b-314232fdc8dc
lower_lam_ci = getproperty.(postmean_ci_lam, :lower)

# ╔═╡ 7028abde-4bd6-11eb-21a9-1fc894463402
upper_lam_ci = getproperty.(postmean_ci_lam, :upper)

# ╔═╡ 797941a8-4bd6-11eb-2beb-27b403e207f6
md"""## Plot confidence intervals (Fig.2b)"""

# ╔═╡ 90c72758-4bd6-11eb-2b50-61600eb25e64
postmean_plot = begin
plot(0:20, upper_lam_ci, fillrange=lower_lam_ci ,seriestype=:sticks,
            frame=:box,
            grid=nothing,
            xguide = L"z",
            yguide = L"\EE{\mu \mid Z=z}",
            legend = :topleft,
            linewidth=2,
            linecolor=:blue,
            alpha = 0.4,
            background_color_legend = :transparent,
            foreground_color_legend = :transparent, ylim=(-0.01,1.01), thickness_scaling=1.3,
            label="Amari",
            size=(500,350))

plot!(0:20, [lower_chisq_ci upper_chisq_ci], seriestype=:scatter,  markershape=:hline,
            label=[L"\chi^2\textrm{-F-Loc}" nothing], markerstrokecolor= :darkorange, markersize=4.5)

plot!(0:20, [lower_dkw_ci upper_dkw_ci], seriestype=:scatter,  markershape=:circle,
             label=["DKW-F-Loc" nothing], color=:black, alpha=0.9, markersize=2.0, markerstrokealpha=0)

plot!([0;20], [0.0; 1.0], seriestype=:line, linestyle=:dot, label=nothing, color=:lightgrey)
end

# ╔═╡ f75f3898-4bd6-11eb-0c18-1bcfedf36254
savefig(postmean_plot, "lord_cressie_posterior_mean.tikz")

# ╔═╡ Cell order:
# ╟─7cd14ada-4bd1-11eb-137b-0591983ae782
# ╠═33cbdf58-4bd1-11eb-2c37-81d5f476828e
# ╠═b852c54c-4bd2-11eb-27c9-6fbfa3dc09e7
# ╠═bd3d470e-4bd1-11eb-2b6a-3342f8a8a272
# ╠═1e6e4c8a-4bd2-11eb-29ae-9bcb1d13a026
# ╟─52bbfa58-4bd4-11eb-0f3b-1977a2671524
# ╠═7fcce91e-4bd2-11eb-3ba0-893699bbb1d6
# ╠═a197f656-4bd2-11eb-1d34-15acb373e216
# ╠═39c4faae-4bd4-11eb-2de6-0fb52b666531
# ╟─7e1c518e-4bd4-11eb-0b09-75616ed63a2e
# ╠═d3f6af54-8161-11eb-0dd9-41ae82c23455
# ╠═caaf9fce-4bd4-11eb-3c4e-318bd5bc838b
# ╠═955f125c-4bd2-11eb-2e6d-195f45c0fb6d
# ╠═98836b02-4bd4-11eb-3f00-f118ee2aa731
# ╠═9dd42a9c-4bd4-11eb-0c4a-f59835f20ed0
# ╟─db6d0e50-4bd4-11eb-2ba1-5349d9a0ad62
# ╠═a51a7982-4bd4-11eb-09b5-434543d65565
# ╠═aa690a8e-4bd4-11eb-3f7c-c36955f21c08
# ╠═c24e0e60-4bd4-11eb-048c-0b02d946fc95
# ╠═14790a00-4bd5-11eb-395e-d37b63309906
# ╠═25dbbcf2-4bd5-11eb-10a9-09afccc8faf7
# ╠═2e7825ee-4bd5-11eb-0433-a30a514b316a
# ╟─e64a9d2e-4bd4-11eb-10bc-4902fa5a6aed
# ╠═ecca61e8-4bd4-11eb-1f1d-5f7a677e157c
# ╠═0316c28e-4bd5-11eb-00c2-2150f66f1721
# ╠═b7f27c2a-4bd5-11eb-3b2b-63532c39571b
# ╠═be60c940-4bd5-11eb-0972-377395236cff
# ╟─c26c3fb0-4bd5-11eb-3607-05043b4df440
# ╠═dec4fae4-4bd5-11eb-337d-0152aba1d793
# ╠═64050d52-4bd6-11eb-2fa6-8bf4fbee5ba3
# ╠═6d844faa-4bd6-11eb-236b-314232fdc8dc
# ╠═7028abde-4bd6-11eb-21a9-1fc894463402
# ╟─797941a8-4bd6-11eb-2beb-27b403e207f6
# ╠═90c72758-4bd6-11eb-2b50-61600eb25e64
# ╠═f75f3898-4bd6-11eb-0c18-1bcfedf36254
