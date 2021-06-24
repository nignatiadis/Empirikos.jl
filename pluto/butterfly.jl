using Distributions
using Empirikos
using MosekTools
using Plots
using LaTeXStrings

begin
	pgfplotsx()
	empty!(PGFPlotsX.CUSTOM_PREAMBLE)
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
	push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}")
end;

theme(:default;
      background_color_legend = :transparent,
      foreground_color_legend = :transparent,
      grid=nothing,
      frame=:box,
      legendfonthalign = :left,
      thickness_scaling=1.3)


Zs = Butterfly.ebayes_samples()
targets = PosteriorMean.(TruncatedPoissonSample.(1:21))

𝒢 = DiscretePriorClass(0.01:0.01:30)

floc_method_dkw = FLocalizationInterval(
							flocalization = DvoretzkyKieferWolfowitz(0.05),
                            convexclass = 𝒢, solver=Mosek.Optimizer)


dkw_floc_cis = confint.(floc_method_dkw, targets, Zs)

discr = integer_discretizer(1:21)

amari = Empirikos.AMARI(convexclass = 𝒢,
                        flocalization = Empirikos.DvoretzkyKieferWolfowitz(0.01),
                        solver=Mosek.Optimizer,
                        discretizer=discr,
                        modulus_model=Empirikos.ModulusModelWithF
                        )


amari_cis = confint.(amari, targets, Zs)


plot(dkw_floc_cis, fillcolor=:darkorange,
    fillalpha=0.5, label="DKW-F-Loc",
    xguide=L"z", yguide=L"\EE{\mu \mid Z=z}" )
plot!(amari_cis, label="AMARI", show_ribbon=true, fillcolor=:blue, fillalpha=0.4)
plot!([1;21], [1;21], color=:black, linestyle=:dash, label="Zipf")


savefig("amari_butterfly.tikz")
