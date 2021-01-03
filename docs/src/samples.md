# EBayesSamples

The design choice of this package, is that each sample is wrapped in a type that represents its likelihood. This works well, since in the empirical Bayes problem, we typically impose (simple) assumptions on the distribution of $Z_i \mid \mu_i$ and complexity emerges from making compound or nonparametric assumptions on the $\mu_i$ and sharing information across $i$. The main advantage is that it then makes it easy to add new likelihoods and have it automatically integrate with the rest of the package (say the nonparametric maximum likelihood estimator) through Julia's multiple dispatch. 

The abstract type is 

```@docs
Empirikos.EBayesSample
```

## StandardNormalSample
We explain the interface in the most well-studied empirical Bayes setting, namely the Gaussian compound decision problem wherein $Z_i \mid \mu_i \sim \mathcal{N}(\mu_i,1)$.  Such a sample is represented through the `StandardNormalSample` type:

```@docs
StandardNormalSample
```

The type can be used in three ways. First, say we observe $Z_i=1.0$, then we reprent that as `Z = StandardNormalSample(1.0)`.  Two more advanced functionalities consist of `StandardNormalSample(missing)`, which represents the random variable $Z_i$ without having observed its realization yet. Finally, `StandardNormalSample(Interval(0.0,1.0))` represents a $Z_i$ whose realization lies in $[0,1]$; this is useful to conduct rigorous discretizations (that can speed up many estimation algorithms). We note that open, closed, unbounded intervals and so forth are allowed, cf. the intervals in the [Intervals.jl](https://github.com/invenia/Intervals.jl) package.

The main interface functions are the following:

```@docs
likelihood_distribution
response
marginalize
```

## Available EBayes sample types

Currently, the following samples have been implemented.

```@docs
NormalSample
BinomialSample
PoissonSample
```

