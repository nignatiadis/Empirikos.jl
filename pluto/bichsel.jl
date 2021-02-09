### A Pluto.jl notebook ###
# v0.12.20

using Markdown
using InteractiveUtils

# ╔═╡ de519962-6a22-11eb-3d6e-0bffe1dcb58a
begin
	using DataFrames
	using Empirikos
	using MosekTools
	using JuMP
	using LaTeXStrings
	using Plots
	pgfplotsx()
end

# ╔═╡ 6dd10bf4-6a23-11eb-3396-eb83889351c4
md"## Predicting automobile insurance claims"

# ╔═╡ 800398b4-6a23-11eb-360c-1b63b21bf2e1
quiet_mosek = optimizer_with_attributes(Mosek.Optimizer,
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP" => 10^(-15))

# ╔═╡ 8b34e620-6a23-11eb-10fb-3bc3cc91d08a
Zs_keys = [PoissonSample.(0:4); PoissonSample(Interval(5,nothing))];

# ╔═╡ a92421aa-6a23-11eb-3540-5bdc4788c0f5
Ns = [103704; 14075; 1766; 255; 45; 8]

# ╔═╡ fb2ad638-6a23-11eb-2090-ada2c0aecbcf
Zs = Empirikos.MultinomialSummary(Zs_keys, Ns)

# ╔═╡ a478a4fe-6a24-11eb-2977-fb2b5bfc3777
postmean_targets = PosteriorMean.(PoissonSample.(0:4))

# ╔═╡ 8296520a-6a24-11eb-0eae-d713be4fe37b
gcal = DiscretePriorClass(0.0:0.01:5.0)

# ╔═╡ 7fc2212a-6a25-11eb-0c9a-5376243ea49a
md"### Compute NPMLE prior and plug-in posterior mean estimates"

# ╔═╡ 8c025154-6a24-11eb-040e-f17b8c9542b0
npmle_fit = fit(NPMLE(gcal, quiet_mosek), Zs)

# ╔═╡ b394759e-6a24-11eb-39c1-c7ccb51a35df
plot(support(npmle_fit.prior), probs(npmle_fit.prior), seriestype=:sticks,
     xlab=L"\mu", ylab=L"G(\mu)",label="")

# ╔═╡ ccec26ac-6a24-11eb-22e8-cdc2213dfccf
npmle_postmeans = postmean_targets.(npmle_fit)

# ╔═╡ 94ac178a-6a25-11eb-0bff-b783cffba0a7
md"### χ² F-localization intervals"

# ╔═╡ 71b632ce-6a25-11eb-0308-475b647cc257
chisq_floc = Empirikos.ChiSquaredFLocalization(0.05)

# ╔═╡ 788609c6-6a25-11eb-177c-7feb1c426574
floc_method_chisq = FLocalizationInterval(flocalization = chisq_floc,
                                       convexclass= gcal, solver=quiet_mosek)

# ╔═╡ 7b4c29ba-6a25-11eb-3cf7-a906cdf5a9d5
chisq_cis = confint.(floc_method_chisq, postmean_targets, Zs)

# ╔═╡ b4185b24-6a25-11eb-36d7-4bfeea3b1ef9
md"### AMARI intervals"

# ╔═╡ bc521d32-6a25-11eb-06e5-81664a86a473
amari_chisq = AMARI(
    flocalization = fit(Empirikos.ChiSquaredFLocalization(0.01), Zs),
    solver=quiet_mosek, convexclass=gcal, discretizer=nothing)

# ╔═╡ d74a0246-6a25-11eb-05f9-79366bc3aed7
postmean_ci_amari = confint.(amari_chisq, postmean_targets, Zs)

# ╔═╡ f5fb418c-6a25-11eb-0a75-6d96d3fc03c6
md"### Table of results"

# ╔═╡ 3bb90574-6a26-11eb-387c-676f6c493673
DataFrame(z=0:4, N=Ns[1:5],
	      NPMLE = round.(npmle_postmeans,digits=2),
	      F_loc_lower = round.(getproperty.(chisq_cis, :lower), digits=2),
	      F_loc_upper = round.(getproperty.(chisq_cis, :upper), digits=2),
	      Amari_lower = round.(getproperty.(postmean_ci_amari, :lower), digits=2),
	      Amari_upper = round.(getproperty.(postmean_ci_amari, :upper), digits=2)
)

# ╔═╡ Cell order:
# ╟─6dd10bf4-6a23-11eb-3396-eb83889351c4
# ╠═de519962-6a22-11eb-3d6e-0bffe1dcb58a
# ╠═800398b4-6a23-11eb-360c-1b63b21bf2e1
# ╠═8b34e620-6a23-11eb-10fb-3bc3cc91d08a
# ╠═a92421aa-6a23-11eb-3540-5bdc4788c0f5
# ╠═fb2ad638-6a23-11eb-2090-ada2c0aecbcf
# ╠═a478a4fe-6a24-11eb-2977-fb2b5bfc3777
# ╠═8296520a-6a24-11eb-0eae-d713be4fe37b
# ╟─7fc2212a-6a25-11eb-0c9a-5376243ea49a
# ╠═8c025154-6a24-11eb-040e-f17b8c9542b0
# ╠═b394759e-6a24-11eb-39c1-c7ccb51a35df
# ╠═ccec26ac-6a24-11eb-22e8-cdc2213dfccf
# ╟─94ac178a-6a25-11eb-0bff-b783cffba0a7
# ╠═71b632ce-6a25-11eb-0308-475b647cc257
# ╠═788609c6-6a25-11eb-177c-7feb1c426574
# ╠═7b4c29ba-6a25-11eb-3cf7-a906cdf5a9d5
# ╟─b4185b24-6a25-11eb-36d7-4bfeea3b1ef9
# ╠═bc521d32-6a25-11eb-06e5-81664a86a473
# ╠═d74a0246-6a25-11eb-05f9-79366bc3aed7
# ╟─f5fb418c-6a25-11eb-0a75-6d96d3fc03c6
# ╠═3bb90574-6a26-11eb-387c-676f6c493673
