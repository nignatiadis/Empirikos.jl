# Empirikos.jl

Consider $n$ independent samples $Z_i$ drawn from the following hierarchical model
```math
\mu_i \sim G, \ \ Z_i \sim p_i(\cdot \mid \mu).
```
Here $G$ is the unknown prior (effect size distribution) and $p_i(\cdot \mid \mu),i=1,\dotsc,n$ are known likelihood functions.

This package provides a unified framework for estimation and inference under the above setting, which is known as the empirical Bayes problem [[robbins1956empirical](@cite)].


## Installation

The package is available from the Julia registry. It may be installed on Julia version 1.5 as follows:
```julia
using Pkg
Pkg.add("Empirikos")
```

## Getting started

Below are some vignettes using this package for empirical Bayes tasks. There are also available as [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks at the following [directory](https://github.com/nignatiadis/Empirikos.jl/tree/master/pluto).

* Nonparametric estimation using the Nonparametric Maximum Likelihood estimator (NPMLE):
  * A [vignette](http://htmlpreview.github.io/?https://github.com/nignatiadis/Empirikos.jl/blob/Pluto/REBayes.jl.html) that partially reproduces the vignette of the REBayes package [[koenker2017rebayes](@cite)]. 
* Nonparametric confidence intervals for empirical Bayes estimands as developed in [[ignatiadis2019bias](@cite)]:
  * [Posterior mean and local false sign rate in a Gaussian dataset.](http://htmlpreview.github.io/?https://github.com/nignatiadis/Empirikos.jl/blob/Pluto/prostate.jl.html)
  * [Posterior mean in a Binomial dataset.](http://htmlpreview.github.io/?https://github.com/nignatiadis/Empirikos.jl/blob/Pluto/lord_cressie.jl.html)
  * [Posterior mean in a Poisson dataset.](http://htmlpreview.github.io/?https://github.com/nignatiadis/Empirikos.jl/blob/Pluto/bichsel.jl.html)



!!! note "Modularity"
      This package has been designed with the goal of modularity. 
      Specialized code (using Julia's multiple dispatch) handles different combinations of estimation targets, statistical algorithms, classes of priors and likelihoods. Please open an issue if there is a combination thereof that you would like to use (and which does not work currently).

      
## Related packages

### In R:

* [REBayes](https://cran.r-project.org/web/packages/REBayes/index.html)  [[koenker2017rebayes](@cite)]. 
* [Ashr](https://cran.r-project.org/web/packages/ashr/index.html)  [[stephens2016false](@cite)]
* [DeconvolveR](https://cran.r-project.org/web/packages/deconvolveR/index.html)  [[narasimhan2020deconvolver](@cite)]
* [EbayesThresh](https://cran.r-project.org/web/packages/EbayesThresh/index.html)  [[johnstone2005ebayesthresh](@cite)]

### In Julia:

* [Aurora.jl](https://github.com/nignatiadis/Aurora.jl)

## References

```@bibliography
```
