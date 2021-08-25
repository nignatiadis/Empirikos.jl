# Empirikos.jl

Consider $n$ independent samples $Z_i$ drawn from the following hierarchical model
```math
\mu_i \sim G, \ \ Z_i \sim p_i(\cdot \mid \mu).
```
Here $G$ is the unknown prior (effect size distribution) and $p_i(\cdot \mid \mu),i=1,\dotsc,n$ are known likelihood functions.

This package provides a unified framework for estimation and inference under the above setting, which is known as the empirical Bayes problem [[robbins1956empirical](@cite)].


## Installation

The package is available from the Julia registry. It may be installed on Julia version 1.6 as follows:
```julia
using Pkg
Pkg.add("Empirikos")
```

For some of its functionality, this package requires a convex programming solver. The requirement for such a solver is that it can solve second order conic programs (SOCP), that it returns the dual variables associated with the SOCP constraints and that it is [supported by JuMP.jl](https://jump.dev/JuMP.jl/dev/installation/#Supported-solvers). We recommend using the [MOSEK](https://www.mosek.com/) solver through the [MosekTools.jl](https://github.com/jump-dev/MosekTools.jl) package and we used MOSEK for all simulations and empirical examples in [[ignatiadis2019bias](@cite)]. MOSEK is a commercial solver, but provides free academic licenses. An open-source alternative is [Hypatia.jl](https://github.com/chriscoey/Hypatia.jl).


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
      Specialized code (using Julia's multiple dispatch) can be easily added to more efficiently handle different combinations of estimation targets, statistical algorithms, classes of priors and likelihoods. Please open an issue if there is a combination thereof that you would like to use (and which does not work currently or is slow).

      
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
