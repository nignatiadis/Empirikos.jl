# Empirikos.jl

Consider $n$ independent samples $Z_i$ drawn from the following hierarchical model
```math
\mu_i \sim G, \ \ Z_i \sim p_i(\cdot \mid \mu).
```
Here $G$ is the unknown prior (effect size distribution) and $p_i(\cdot \mid \mu),i=1,\dotsc,n$ are known likelihood functions.

This package provides a unified framework for estimation and inference under the above setting, which is known as the empirical Bayes problem [[robbins1956empirical](@cite)].


!!! note "Modularity"
      The package here has been designed with the goal of modularity. 
      Specialized code (using Julia's multiple dispatch) handles different combinations of estimation targets, statistical algorithms, classes of priors and likelihoods. Please open an issue if there is a combination thereof that you would like to use (and which does not work currently).


## Installation

The package is available from the Julia registry. It may be installed as follows:
```julia
using Pkg
Pkg.add("Empirikos")
```

## Related packages in R


* [REBayes](https://cran.r-project.org/web/packages/REBayes/index.html)  [[koenker2017rebayes](@cite)]. A partial reproduction of the REBayes vignette with `Empirikos.jl` is available [here](http://htmlpreview.github.io/?https://github.com/nignatiadis/Empirikos.jl/blob/Pluto/REBayes.jl.html).
* [Ashr](https://cran.r-project.org/web/packages/ashr/index.html)  [[stephens2016false](@cite)]
* [DeconvolveR](https://cran.r-project.org/web/packages/deconvolveR/index.html)  [[narasimhan2020deconvolver](@cite)]
* [EbayesThresh](https://cran.r-project.org/web/packages/EbayesThresh/index.html)  [[johnstone2005ebayesthresh](@cite)]


## References

```@bibliography
```
