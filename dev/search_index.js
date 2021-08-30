var documenterSearchIndex = {"docs":
[{"location":"flocalizations/#F-Localizations","page":"F-Localizations","title":"F-Localizations","text":"","category":"section"},{"location":"flocalizations/","page":"F-Localizations","title":"F-Localizations","text":"Empirikos.FLocalization\nEmpirikos.FittedFLocalization","category":"page"},{"location":"flocalizations/#Empirikos.FLocalization","page":"F-Localizations","title":"Empirikos.FLocalization","text":"Abstract type representing F-Localizations.\n\n\n\n\n\n","category":"type"},{"location":"flocalizations/#Empirikos.FittedFLocalization","page":"F-Localizations","title":"Empirikos.FittedFLocalization","text":"Abstract type representing a fitted F-Localization    (i.e., wherein the F-localization has already been determined by data).\n\n\n\n\n\n","category":"type"},{"location":"flocalizations/#DKW-F-Localization","page":"F-Localizations","title":"DKW-F-Localization","text":"","category":"section"},{"location":"flocalizations/","page":"F-Localizations","title":"F-Localizations","text":"DvoretzkyKieferWolfowitz","category":"page"},{"location":"flocalizations/#Empirikos.DvoretzkyKieferWolfowitz","page":"F-Localizations","title":"Empirikos.DvoretzkyKieferWolfowitz","text":"DvoretzkyKieferWolfowitz(;α = 0.05, max_constraints = 1000) <: FLocalization\n\nThe Dvoretzky-Kiefer-Wolfowitz band (based on the Kolmogorov-Smirnov distance) at confidence level 1-α that bounds the distance of the true distribution function to the ECDF widehatF_n based on n samples. The constant of the band is the sharp constant derived by Massart:\n\nF text distribution  sup_t in mathbb Rlvert F(t) - widehatF_n(t) rvert  leq  sqrtlog(2alpha)(2n)\n\nThe supremum above is enforced discretely on at most max_constraints number of points.\n\n\n\n\n\n","category":"type"},{"location":"flocalizations/#\\chi2-F-Localization","page":"F-Localizations","title":"chi^2-F-Localization","text":"","category":"section"},{"location":"flocalizations/","page":"F-Localizations","title":"F-Localizations","text":"ChiSquaredFLocalization","category":"page"},{"location":"flocalizations/#Empirikos.ChiSquaredFLocalization","page":"F-Localizations","title":"Empirikos.ChiSquaredFLocalization","text":"ChiSquaredFLocalization(α) <: FLocalization\n\nThe chi^2 F-localization at confidence level 1-alpha for a discrete random variable taking values in 0dotsc N. It is equal to:\n\nf sum_x=0^N frac(n hatf_n(x) - n f(x))^2n f(x) leq chi^2_N1-alpha\n\nwhere chi^2_N1-alpha is the 1-alpha quantile of the Chi-squared distribution with N degrees of freedom, n is the sample size, hatf_n(x) is the proportion of samples equal to x and f(x) is then population pmf.\n\n\n\n\n\n","category":"type"},{"location":"flocalizations/#Gauss-F-Localization","page":"F-Localizations","title":"Gauss-F-Localization","text":"","category":"section"},{"location":"flocalizations/","page":"F-Localizations","title":"F-Localizations","text":"Empirikos.InfinityNormDensityBand","category":"page"},{"location":"flocalizations/#Empirikos.InfinityNormDensityBand","page":"F-Localizations","title":"Empirikos.InfinityNormDensityBand","text":"InfinityNormDensityBand(;a_min,\n                         a_max,\n                         kernel  =  Empirikos.FlatTopKernel(),\n                         bootstrap = :Multinomial,\n                         nboot = 1000,\n                         α = 0.05,\n                         rng = Random.MersenneTwister(1)\n                    )  <: FLocalization\n\nThis struct contains hyperparameters that will be used for constructing a neighborhood of the marginal density. The steps of the method (and corresponding hyperparameter meanings) are as follows\n\nFirst a kernel density estimate barf with kernel is fit to the data.\nSecond, a bootstrap (options: :Multinomial or Poisson) with nboot bootstrap replicates will be used to estimate c_n, such that:\n\nliminf_n to inftymathbbPleftsup_x in a_textmin  a_textmax  barf(x) - f(x) leq c_ nright geq 1-alpha\n\nNote that the bound is valid from a_min to a_max. α is the nominal level and finally rng sets the seed for the bootstrap samples.\n\n\n\n\n\n","category":"type"},{"location":"flocalizations/","page":"F-Localizations","title":"F-Localizations","text":"This F-Localization currently only works for homoskedastic Normal samples with common noise variance sigma^2. By default the above uses the following kernel, with bandwidth h = sigmasqrtlog(n), where n is the sample size:","category":"page"},{"location":"flocalizations/","page":"F-Localizations","title":"F-Localizations","text":"Empirikos.FlatTopKernel\nEmpirikos.SincKernel","category":"page"},{"location":"flocalizations/#Empirikos.FlatTopKernel","page":"F-Localizations","title":"Empirikos.FlatTopKernel","text":"FlatTopKernel(h) < InfiniteOrderKernel\n\nImplements the FlatTopKernel with bandwidth h to be used for kernel density estimation through the KernelDensity.jl package. The flat-top kernel is defined as follows:\n\nK(x) = fracsin^2(11x2)-sin^2(x2)pi x^2 20\n\nIts use case is similar to the SincKernel, however it has the advantage of being integrable (in the Lebesgue sense) and having bounded total variation. Its Fourier transform is the following:\n\nK^*(t) = begincases\n 1  text if  tleq 1 \n 0 text if  t geq 11 \n 11-10t text if  t in 111\n endcases\n\njulia> Empirikos.FlatTopKernel(0.1)\nFlatTopKernel | bandwidth = 0.1\n\n\n\n\n\n","category":"type"},{"location":"flocalizations/#Empirikos.SincKernel","page":"F-Localizations","title":"Empirikos.SincKernel","text":"SincKernel(h) <: InfiniteOrderKernel\n\nImplements the SincKernel with bandwidth h to be used for kernel density estimation through the KernelDensity.jl package. The sinc kernel is defined as follows:\n\nK_textsinc(x) = fracsin(x)pi x\n\nIt is not typically used for kernel density estimation, because this kernel is not a density itself. However, it is particularly well suited to deconvolution problems and estimation of very smooth densities because its Fourier transform is the following:\n\nK^*_textsinc(t) = mathbf 1( t in -11)\n\n\n\n\n\n","category":"type"},{"location":"estimands/#Estimands","page":"Estimands","title":"Estimands","text":"","category":"section"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"This package defines dedicated types to describe empirical Bayes estimands (that can be used for estimation or inference).","category":"page"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Empirikos.EBayesTarget","category":"page"},{"location":"estimands/#Empirikos.EBayesTarget","page":"Estimands","title":"Empirikos.EBayesTarget","text":"Abstract type that describes Empirical Bayes estimands (which we want to estimate or conduct inference for).\n\n\n\n\n\n","category":"type"},{"location":"estimands/#Example:-PosteriorMean","page":"Estimands","title":"Example: PosteriorMean","text":"","category":"section"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"PosteriorMean","category":"page"},{"location":"estimands/#Empirikos.PosteriorMean","page":"Estimands","title":"Empirikos.PosteriorMean","text":"PosteriorMean(Z::EBayesSample) <: AbstractPosteriorTarget\n\nType representing the posterior mean, i.e.,\n\nE_Gmu_i mid Z_i = z\n\n\n\n\n\n","category":"type"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"A target::EBayesTarget, such as PosteriorMean, may be used as a callable on distributions (priors).","category":"page"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"julia> G = Normal()\nNormal{Float64}(μ=0.0, σ=1.0)\n\njulia> postmean1 = PosteriorMean(StandardNormalSample(1.0))\nPosteriorMean{StandardNormalSample{Float64}}(Z=     1.0 | σ=1.0  )\njulia> postmean1(G)\n0.5\n\njulia> postmean2 = PosteriorMean(NormalSample(1.0, sqrt(3.0)))\nPosteriorMean{NormalSample{Float64,Float64}}(Z=     1.0 | σ=1.732)\njulia> postmean2(G)\n0.25000000000000006","category":"page"},{"location":"estimands/#Posterior-estimands","page":"Estimands","title":"Posterior estimands","text":"","category":"section"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"In addition to PosteriorMean, other implemented posterior estimands are the following:","category":"page"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Empirikos.PosteriorProbability\nPosteriorVariance","category":"page"},{"location":"estimands/#Empirikos.PosteriorProbability","page":"Estimands","title":"Empirikos.PosteriorProbability","text":"PosteriorProbability(Z::EBayesSample, s) <: AbstractPosteriorTarget\n\nType representing the posterior probability, i.e.,\n\nProb_Gmu_i in s mid Z_i = z\n\n\n\n\n\n","category":"type"},{"location":"estimands/#Empirikos.PosteriorVariance","page":"Estimands","title":"Empirikos.PosteriorVariance","text":"PosteriorVariance(Z::EBayesSample) <: AbstractPosteriorTarget\n\nType representing the posterior variance, i.e.,\n\nV_Gmu_i mid Z_i = z\n\n\n\n\n\n","category":"type"},{"location":"estimands/#Linear-functionals","page":"Estimands","title":"Linear functionals","text":"","category":"section"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"A special case of empirical Bayes estimands are linear functionals:","category":"page"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Empirikos.LinearEBayesTarget","category":"page"},{"location":"estimands/#Empirikos.LinearEBayesTarget","page":"Estimands","title":"Empirikos.LinearEBayesTarget","text":"LinearEBayesTarget <: EBayesTarget\n\nAbstract type that describes Empirical Bayes estimands that are linear functionals of the prior G.\n\n\n\n\n\n","category":"type"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Currently available linear functionals:","category":"page"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Empirikos.PriorDensity\nEmpirikos.MarginalDensity","category":"page"},{"location":"estimands/#Empirikos.PriorDensity","page":"Estimands","title":"Empirikos.PriorDensity","text":"PriorDensity(z::Float64) <: LinearEBayesTarget\n\nExample call\n\nPriorDensity(2.0)\n\nDescription\n\nThis is the evaluation functional of the density of G at z, i.e., L(G) = G(z) = g(z) or in Julia code L(G) = pdf(G, z).\n\n\n\n\n\n","category":"type"},{"location":"estimands/#Empirikos.MarginalDensity","page":"Estimands","title":"Empirikos.MarginalDensity","text":"MarginalDensity(Z::EBayesSample) <: LinearEBayesTarget\n\nExample call\n\nMarginalDensity(StandardNormalSample(2.0))\n\nDescription\n\nDescribes the marginal density evaluated at Z=z  (e.g. Z=2 in the example above). In the example above the sample is drawn from the hierarchical model\n\nmu sim G Z sim mathcalN(01)\n\nIn other words, letting varphi the Standard Normal pdf\n\nL(G) = varhi star dG(z)\n\nNote that 2.0 has to be wrapped inside StandardNormalSample(2.0) since this target depends not only on G and the location, but also on the likelihood.\n\n\n\n\n\n","category":"type"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Posterior estimands such as PosteriorMean can be typically decomposed into two linear functionals, a numerator' and adenominator':","category":"page"},{"location":"estimands/","page":"Estimands","title":"Estimands","text":"Base.numerator(::Empirikos.AbstractPosteriorTarget)\nBase.denominator(::Empirikos.AbstractPosteriorTarget)","category":"page"},{"location":"estimands/#Base.numerator-Tuple{Empirikos.AbstractPosteriorTarget}","page":"Estimands","title":"Base.numerator","text":"Base.numerator(target::AbstractPosteriorTarget)\n\nSuppose a posterior target theta_G(z), such as the posterior mean can be written as:\n\ntheta_G(z) = frac a_G(z)f_G(z) = frac int h(mu)dG(mu)int p(z mid mu)dG(mu)\n\nFor example, for the posterior mean h(mu) =  mu cdot p(z mid mu). Then Base.numerator returns the linear functional representing G mapsto a_G(z).\n\n\n\n\n\n","category":"method"},{"location":"estimands/#Base.denominator-Tuple{Empirikos.AbstractPosteriorTarget}","page":"Estimands","title":"Base.denominator","text":"Base.denominator(target::AbstractPosteriorTarget)\n\nSuppose a posterior target theta_G(z), such as the posterior mean can be written as:\n\ntheta_G(z) = frac a_G(z)f_G(z) = frac int h(mu)dG(mu)int p(z mid mu)dG(mu)\n\nFor example, for the posterior mean h(mu) =  mu cdot p(z mid mu). Then Base.denominator returns the linear functional representing G mapsto f_G(z) (i.e., typically the marginal density). Also see Base.numerator(::AbstractPosteriorTarget).\n\n\n\n\n\n","category":"method"},{"location":"intervals/#Confidence-intervals","page":"Confidence intervals","title":"Confidence intervals","text":"","category":"section"},{"location":"intervals/","page":"Confidence intervals","title":"Confidence intervals","text":"Here we describe two methods for forming confidence intervals for empirical Bayes estimands (Empirikos.EBayesTarget).","category":"page"},{"location":"intervals/#F-Localization-based-intervals","page":"Confidence intervals","title":"F-Localization based intervals","text":"","category":"section"},{"location":"intervals/","page":"Confidence intervals","title":"Confidence intervals","text":"FLocalizationInterval","category":"page"},{"location":"intervals/#Empirikos.FLocalizationInterval","page":"Confidence intervals","title":"Empirikos.FLocalizationInterval","text":"FLocalizationInterval(flocalization::Empirikos.FLocalization,\n                      convexclass::Empirikos.ConvexPriorClass,\n                      solver,\n                      n_bisection = 100)\n\nMethod for computing frequentist confidence intervals for empirical Bayes estimands. Here flocalization is a  Empirikos.FLocalization, convexclass is a Empirikos.ConvexPriorClass, solver is a JuMP.jl compatible solver.\n\nn_bisection is relevant only for combinations of flocalization and convexclass for     which the Charnes-Cooper transformation is not applicable/implemented.     Instead, a quasi-convex optimization problem is solved by bisection and     increasing n_bisection increases     accuracy (at the cost of more computation).\n\n\n\n\n\n","category":"type"},{"location":"intervals/#AMARI-intervals","page":"Confidence intervals","title":"AMARI intervals","text":"","category":"section"},{"location":"intervals/","page":"Confidence intervals","title":"Confidence intervals","text":"AMARI","category":"page"},{"location":"intervals/#Empirikos.AMARI","page":"Confidence intervals","title":"Empirikos.AMARI","text":"AMARI(convexclass::Empirikos.ConvexPriorClass,\n      flocalization::Empirikos.FLocalization,\n      solver,\n      plugin_G = KolmogorovSmirnovMinimumDistance(convexclass, solver))\n\nAffine Minimax Anderson-Rubin intervals for empirical Bayes estimands. Here flocalization is a  pilot Empirikos.FLocalization, convexclass is a Empirikos.ConvexPriorClass, solver is a JuMP.jl compatible solver. plugin_G is a Empirikos.EBayesMethod used as an initial estimate of the marginal distribution of the i.i.d. samples Z.\n\n\n\n\n\n","category":"type"},{"location":"intervals/#Interface","page":"Confidence intervals","title":"Interface","text":"","category":"section"},{"location":"intervals/","page":"Confidence intervals","title":"Confidence intervals","text":"StatsBase.confint(::AMARI, ::Empirikos.AbstractPosteriorTarget, ::Any)","category":"page"},{"location":"estimation/#Prior-Estimation","page":"Prior Estimation","title":"Prior Estimation","text":"","category":"section"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"Empirikos.EBayesMethod","category":"page"},{"location":"estimation/#Empirikos.EBayesMethod","page":"Prior Estimation","title":"Empirikos.EBayesMethod","text":"Abstract type representing empirical Bayes estimation methods.\n\n\n\n\n\n","category":"type"},{"location":"estimation/#Nonparametric-estimation","page":"Prior Estimation","title":"Nonparametric estimation","text":"","category":"section"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"The typical call for estimating the prior G based on empirical Bayes samples Zs is the following,","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"StatsBase.fit(method, Zs)","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"Above, method is a type that specifies both the assumptions made on G (say, the convex prior class mathcalG in which G lies), as well as details concerning the computation (typically a JuMP.jl compatible convex programming solver). ","category":"page"},{"location":"estimation/#Nonparametric-Maximum-Likelihood-estimation-(NPMLE)","page":"Prior Estimation","title":"Nonparametric Maximum Likelihood estimation (NPMLE)","text":"","category":"section"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"For example, let us consider the nonparametric maximum likelihood estimator:","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"NPMLE","category":"page"},{"location":"estimation/#Empirikos.NPMLE","page":"Prior Estimation","title":"Empirikos.NPMLE","text":"NPMLE(convexclass, solver) <: Empirikos.EBayesMethod\n\nGiven n independent samples Z_i from the empirical Bayes problem with prior G known to lie in the convexclass mathcalG, estimate G by Nonparametric Maximum Likelihood (NPMLE)\n\nwidehatG_n in operatornameargmax_G in mathcalGleftsum_i=1^n log( f_iG(Z_i)) right\n\nwhere f_iG(z) = int p_i(z mid mu) dG(mu) is the marginal density of the i-th sample. The optimization is conducted by a JuMP compatible solver.\n\n\n\n\n\n","category":"type"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"Suppose we have Poisson samples Zs, each with a different mean mu_i drawn from G=U15:","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"using Distributions\nn = 1000\nμs = rand(Uniform(1,5), n)\nZs = PoissonSample.(rand.(Poisson.(μs)))","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"We can then estimate G as follows using Mosek:","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"using MosekTools\ng_hat = fit(NPMLE(DiscretePriorClass(), Mosek.Optimizer), Zs)","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"Or we can use the open-source Hypatia.jl solver:","category":"page"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"using Hypatia\ng_hat = fit(NPMLE(DiscretePriorClass(), Hypatia.Optimizer), Zs)","category":"page"},{"location":"estimation/#Other-available-nonparametric-methods","page":"Prior Estimation","title":"Other available nonparametric methods","text":"","category":"section"},{"location":"estimation/","page":"Prior Estimation","title":"Prior Estimation","text":"Empirikos.KolmogorovSmirnovMinimumDistance","category":"page"},{"location":"estimation/#Empirikos.KolmogorovSmirnovMinimumDistance","page":"Prior Estimation","title":"Empirikos.KolmogorovSmirnovMinimumDistance","text":"KolmogorovSmirnovMinimumDistance(convexclass, solver) <: Empirikos.EBayesMethod\n\nGiven n i.i.d. samples from the empirical Bayes problem with prior G known to lie in the convexclass mathcalG , estimate G as follows:\n\nwidehatG_n in operatornameargmin_G in mathcalGsup_t in mathbb Rlvert F_G(t) - widehatF_n(t)rvert\n\nwhere widehatF_n is the ECDF of the samples. The optimization is conducted by a JuMP compatible solver.\n\n\n\n\n\n","category":"type"},{"location":"convexpriors/#Convex-prior-classes","page":"Convex prior classes","title":"Convex prior classes","text":"","category":"section"},{"location":"convexpriors/","page":"Convex prior classes","title":"Convex prior classes","text":"The starting point for many empirical Bayes tasks, such as inference or estimation, is to posit that the true prior G lies in a convex class of priors mathcalG. Such classes of priors  are represented in this package through the abstract type,","category":"page"},{"location":"convexpriors/","page":"Convex prior classes","title":"Convex prior classes","text":"Empirikos.ConvexPriorClass","category":"page"},{"location":"convexpriors/#Empirikos.ConvexPriorClass","page":"Convex prior classes","title":"Empirikos.ConvexPriorClass","text":"Abstract type representing convex classes of probability distributions mathcalG.\n\n\n\n\n\n","category":"type"},{"location":"convexpriors/","page":"Convex prior classes","title":"Convex prior classes","text":"Currently, the following choices for mathcalG are available:","category":"page"},{"location":"convexpriors/","page":"Convex prior classes","title":"Convex prior classes","text":"DiscretePriorClass\nMixturePriorClass\nGaussianScaleMixtureClass","category":"page"},{"location":"convexpriors/#Empirikos.DiscretePriorClass","page":"Convex prior classes","title":"Empirikos.DiscretePriorClass","text":"DiscretePriorClass(support) <: Empirikos.ConvexPriorClass\n\nType representing the family of all discrete distributions supported on a subset of support, i.e., it represents all DiscreteNonParametric distributions with support = support and probs taking values on the probability simplex.\n\nNote that DiscretePriorClass(support)(probs) == DiscreteNonParametric(support, probs).\n\nExamples\n\njulia> gcal = DiscretePriorClass([0,0.5,1.0])\nDiscretePriorClass | support = [0.0, 0.5, 1.0]\n\njulia> gcal([0.2,0.2,0.6])\nDiscreteNonParametric{Float64, Float64, Vector{Float64}, Vector{Float64}}(support=[0.0, 0.5, 1.0], p=[0.2, 0.2, 0.6])\n\n\n\n\n\n","category":"type"},{"location":"convexpriors/#Empirikos.MixturePriorClass","page":"Convex prior classes","title":"Empirikos.MixturePriorClass","text":"MixturePriorClass(components) <: Empirikos.ConvexPriorClass\n\nType representing the family of all mixture distributions with mixing components equal to components, i.e., it represents all MixtureModel distributions with components = components and probs taking values on the probability simplex.\n\nNote that MixturePriorClass(components)(probs) == MixtureModel(components, probs).\n\nExamples\n\njulia> gcal = MixturePriorClass([Normal(0,1), Normal(0,2)])\nMixturePriorClass (K = 2)\nNormal{Float64}(μ=0.0, σ=1.0)\nNormal{Float64}(μ=0.0, σ=2.0)\n\njulia> gcal([0.2,0.8])\nMixtureModel{Normal{Float64}}(K = 2)\ncomponents[1] (prior = 0.2000): Normal{Float64}(μ=0.0, σ=1.0)\ncomponents[2] (prior = 0.8000): Normal{Float64}(μ=0.0, σ=2.0)\n\n\n\n\n\n","category":"type"},{"location":"convexpriors/#Empirikos.GaussianScaleMixtureClass","page":"Convex prior classes","title":"Empirikos.GaussianScaleMixtureClass","text":"GaussianScaleMixtureClass(σs) <: Empirikos.ConvexPriorClass\n\nType representing the family of mixtures of Gaussians with mean 0 and standard deviations equal to σs. GaussianScaleMixtureClass(σs) represents the same class of distributions as MixturePriorClass.(Normal.(0, σs))\n\njulia> gcal = GaussianScaleMixtureClass([1.0,2.0])\nGaussianScaleMixtureClass | σs = [1.0, 2.0]\n\njulia> gcal([0.2,0.8])\nMixtureModel{Normal{Float64}}(K = 2)\ncomponents[1] (prior = 0.2000): Normal{Float64}(μ=0.0, σ=1.0)\ncomponents[2] (prior = 0.8000): Normal{Float64}(μ=0.0, σ=2.0)\n\n\n\n\n\n","category":"type"},{"location":"#Empirikos.jl","page":"Introduction","title":"Empirikos.jl","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Consider n independent samples Z_i drawn from the following hierarchical model","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"mu_i sim G   Z_i sim p_i(cdot mid mu)","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Here G is the unknown prior (effect size distribution) and p_i(cdot mid mu)i=1dotscn are known likelihood functions.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"This package provides a unified framework for estimation and inference under the above setting, which is known as the empirical Bayes problem [Herbert Robbins (1956)].","category":"page"},{"location":"#Installation","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"The package is available from the Julia registry. It may be installed on Julia version 1.6 as follows:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"using Pkg\nPkg.add(\"Empirikos\")","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"For some of its functionality, this package requires a convex programming solver. The requirement for such a solver is that it can solve second order conic programs (SOCP), that it returns the dual variables associated with the SOCP constraints and that it is supported by JuMP.jl. We recommend using the MOSEK solver through the MosekTools.jl package and we used MOSEK for all simulations and empirical examples in [Nikolaos Ignatiadis, Stefan Wager (2019+)]. MOSEK is a commercial solver, but provides free academic licenses. An open-source alternative is Hypatia.jl.","category":"page"},{"location":"#Getting-started","page":"Introduction","title":"Getting started","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Below are some vignettes using this package for empirical Bayes tasks. There are also available as Pluto.jl notebooks at the following directory.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Nonparametric estimation using the Nonparametric Maximum Likelihood estimator (NPMLE):\nA vignette that partially reproduces the vignette of the REBayes package [Roger Koenker, Jiaying Gu (2017)]. \nNonparametric confidence intervals for empirical Bayes estimands as developed in [Nikolaos Ignatiadis, Stefan Wager (2019+)]:\nPosterior mean and local false sign rate in a Gaussian dataset.\nPosterior mean in a Binomial dataset.\nPosterior mean in a Poisson dataset.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"note: Modularity\nThis package has been designed with the goal of modularity.    Specialized code (using Julia's multiple dispatch) can be easily added to more efficiently handle different combinations of estimation targets, statistical algorithms, classes of priors and likelihoods. Please open an issue if there is a combination thereof that you would like to use (and which does not work currently or is slow).","category":"page"},{"location":"#Related-packages","page":"Introduction","title":"Related packages","text":"","category":"section"},{"location":"#In-R:","page":"Introduction","title":"In R:","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"REBayes  [Roger Koenker, Jiaying Gu (2017)]. \nAshr  [Matthew Stephens (2016)]\nDeconvolveR  [Balasubramanian Narasimhan, Bradley Efron (2020)]\nEbayesThresh  [Iain M Johnstone, Bernard W Silverman (2005)]","category":"page"},{"location":"#In-Julia:","page":"Introduction","title":"In Julia:","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Aurora.jl","category":"page"},{"location":"#References","page":"Introduction","title":"References","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"","category":"page"},{"location":"samples/#Empirical-Bayes-samples","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"","category":"section"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"The design choice of this package, is that each sample is wrapped in a type that represents its likelihood. This works well, since in the empirical Bayes problem, we typically impose (simple) assumptions on the distribution of Z_i mid mu_i and complexity emerges from making compound or nonparametric assumptions on the mu_i and sharing information across i. The main advantage is that it then makes it easy to add new likelihoods and have it automatically integrate with the rest of the package (say the nonparametric maximum likelihood estimator) through Julia's multiple dispatch. ","category":"page"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"The abstract type is ","category":"page"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"Empirikos.EBayesSample","category":"page"},{"location":"samples/#Empirikos.EBayesSample","page":"Empirical Bayes samples","title":"Empirikos.EBayesSample","text":"EBayesSample{T}\n\nAbstract type representing empirical Bayes samples with realizations of type T.\n\n\n\n\n\n","category":"type"},{"location":"samples/#Example:-StandardNormalSample","page":"Empirical Bayes samples","title":"Example: StandardNormalSample","text":"","category":"section"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"We explain the interface in the most well-studied empirical Bayes setting, namely the Gaussian compound decision problem wherein Z_i mid mu_i sim mathcalN(mu_i1).  Such a sample is represented through the StandardNormalSample type:","category":"page"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"StandardNormalSample","category":"page"},{"location":"samples/#Empirikos.StandardNormalSample","page":"Empirical Bayes samples","title":"Empirikos.StandardNormalSample","text":"StandardNormalSample(Z)\n\nAn observed sample Z drawn from a Normal distribution with known variance sigma^2 =1.\n\nZ sim mathcalN(mu 1)\n\nmu is assumed unknown. The type above is used when the sample Z is to be used for estimation or inference of mu.\n\nStandardNormalSample(0.5)          #Z=0.5\n\n\n\n\n\n","category":"type"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"The type can be used in three ways. First, say we observe Z_i=10, then we reprent that as Z = StandardNormalSample(1.0).  Two more advanced functionalities consist of StandardNormalSample(missing), which represents the random variable Z_i without having observed its realization yet. Finally, StandardNormalSample(Interval(0.0,1.0)) represents a Z_i whose realization lies in 01; this is useful to conduct rigorous discretizations (that can speed up many estimation algorithms). We note that open, closed, unbounded intervals and so forth are allowed, cf. the intervals in the Intervals.jl package.","category":"page"},{"location":"samples/#Interface","page":"Empirical Bayes samples","title":"Interface","text":"","category":"section"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"The main interface functions are the following:","category":"page"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"likelihood_distribution\nresponse(::EBayesSample)\nmarginalize\npdf(::Distribution, ::EBayesSample)\ncdf(::Distribution, ::EBayesSample)\nccdf(::Distribution, ::EBayesSample)","category":"page"},{"location":"samples/#Empirikos.likelihood_distribution","page":"Empirical Bayes samples","title":"Empirikos.likelihood_distribution","text":"likelihood_distribution(Z::EBayesSample, μ::Number)\n\nReturns the distribution p(cdot mid mu) of Z mid mu (the return type being a Distributions.jl Distribution).\n\nExamples\n\njulia> likelihood_distribution(StandardNormalSample(1.0), 2.0)\nNormal{Float64}(μ=2.0, σ=1.0)\n\n\n\n\n\n","category":"function"},{"location":"samples/#StatsBase.response-Tuple{EBayesSample}","page":"Empirical Bayes samples","title":"StatsBase.response","text":"response(Z::EBayesSample{T})\n\nReturns the concrete realization of Z as type T, thus dropping the information about the likelihood.\n\nExamples\n\njulia> response(StandardNormalSample(1.0))\n1.0\n\n\n\n\n\n","category":"method"},{"location":"samples/#Empirikos.marginalize","page":"Empirical Bayes samples","title":"Empirikos.marginalize","text":"marginalize(Z::EBayesSample, prior::Distribution)\n\nGiven a prior distribution G and  EBayesSample Z, return that marginal distribution of Z. Works for EBayesSample{Missing}`, i.e., no realization is needed.\n\nExamples\n\njldoctest julia> marginalize(StandardNormalSample(1.0), Normal(2.0, sqrt(3))) Normal{Float64}(μ=2.0, σ=1.9999999999999998)`\n\n\n\n\n\n","category":"function"},{"location":"samples/#Distributions.pdf-Tuple{Distribution,EBayesSample}","page":"Empirical Bayes samples","title":"Distributions.pdf","text":"pdf(prior::Distribution, Z::EBayesSample)\n\nGiven a prior G and EBayesSample Z, compute the marginal density of Z.\n\nExamples\n\njulia> Z = StandardNormalSample(1.0)\nZ=     1.0 | σ=1.0\njulia> prior = Normal(2.0, sqrt(3))\nNormal{Float64}(μ=2.0, σ=1.7320508075688772)\njulia> pdf(prior, Z)\n0.17603266338214976\njulia> pdf(Normal(2.0, 2.0), 1.0)\n0.17603266338214976\n\n\n\n\n\n","category":"method"},{"location":"samples/#Distributions.cdf-Tuple{Distribution,EBayesSample}","page":"Empirical Bayes samples","title":"Distributions.cdf","text":"cdf(prior::Distribution, Z::EBayesSample)\n\nGiven a prior G and EBayesSample Z, evaluate the CDF of the marginal distribution of Z at response(Z).\n\n\n\n\n\n","category":"method"},{"location":"samples/#Distributions.ccdf-Tuple{Distribution,EBayesSample}","page":"Empirical Bayes samples","title":"Distributions.ccdf","text":"ccdf(prior::Distribution, Z::EBayesSample)\n\nGiven a prior G and EBayesSample Z, evaluate the complementary CDF of the marginal distribution of Z at response(Z).\n\n\n\n\n\n","category":"method"},{"location":"samples/#Other-implemented-EBayesSample-types","page":"Empirical Bayes samples","title":"Other implemented EBayesSample types","text":"","category":"section"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"Currently, the following samples have been implemented.","category":"page"},{"location":"samples/","page":"Empirical Bayes samples","title":"Empirical Bayes samples","text":"NormalSample\nBinomialSample\nPoissonSample","category":"page"},{"location":"samples/#Empirikos.NormalSample","page":"Empirical Bayes samples","title":"Empirikos.NormalSample","text":"NormalSample(Z,σ)\n\nAn observed sample Z drawn from a Normal distribution with known variance sigma^2  0.\n\nZ sim mathcalN(mu sigma^2)\n\nmu is assumed unknown. The type above is used when the sample Z is to be used for estimation or inference of mu.\n\nNormalSample(0.5, 1.0)          #Z=0.5, σ=1\n\n\n\n\n\n","category":"type"},{"location":"samples/#Empirikos.BinomialSample","page":"Empirical Bayes samples","title":"Empirikos.BinomialSample","text":"BinomialSample(Z, n)\n\nAn observed sample Z drawn from a Binomial distribution with n trials.\n\nZ sim textBinomial(n p)\n\np is assumed unknown. The type above is used when the sample Z is to be used for estimation or inference of p.\n\njulia> BinomialSample(2, 10)          # 2 out of 10 trials successful\nZ=2  | n=10\n\n\n\n\n\n","category":"type"},{"location":"samples/#Empirikos.PoissonSample","page":"Empirical Bayes samples","title":"Empirikos.PoissonSample","text":"PoissonSample(Z, E)\n\nAn observed sample Z drawn from a Poisson distribution,\n\nZ sim textPoisson(mu cdot E)\n\nThe multiplying intensity E is assumed to be known (and equal to 1.0 by default), while mu is assumed unknown. The type above is used when the sample Z is to be used for estimation or inference of mu.\n\nPoissonSample(3)\nPoissonSample(3, 1.5)\n\n\n\n\n\n","category":"type"}]
}
