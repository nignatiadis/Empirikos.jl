# Estimation

## Nonparametric estimation

The typical call for estimating the prior $G$ based on empirical Bayes samples `Zs` is the following,

```julia
StatsBase.fit(method, Zs)
```
Above, `method` is a type that specifies both the assumptions made on $G$ (say, the convex prior class $\mathcal{G}$ in which $G$ lies), as well as details concerning the computation (typically a JuMP.jl compatible convex programming solver). The following methods for nonparametric estimation are currently available.

```@docs
NPMLE
Empirikos.KolmogorovSmirnovMinimumDistance
```

