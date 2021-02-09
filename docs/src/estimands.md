# Estimands

This package defines dedicated types to describe empirical Bayes estimands (that can be used for estimation or inference).

```@docs
Empirikos.EBayesTarget
```

## Example: PosteriorMean

```@docs
PosteriorMean
```

A `target::EBayesTarget`, such as `PosteriorMean`, may be used as a callable on distributions (priors).

```julia-repl
julia> G = Normal()
Normal{Float64}(μ=0.0, σ=1.0)

julia> postmean1 = PosteriorMean(StandardNormalSample(1.0))
PosteriorMean{StandardNormalSample{Float64}}(Z=     1.0 | σ=1.0  )
julia> postmean1(G)
0.5

julia> postmean2 = PosteriorMean(NormalSample(1.0, sqrt(3.0)))
PosteriorMean{NormalSample{Float64,Float64}}(Z=     1.0 | σ=1.732)
julia> postmean2(G)
0.25000000000000006
```


## Posterior estimands 

In addition to [`PosteriorMean`](@ref), other implemented posterior estimands are the following:

```@docs
Empirikos.PosteriorProbability
PosteriorVariance
```

## Linear functionals 

A special case of empirical Bayes estimands are linear functionals:
```@docs
Empirikos.LinearEBayesTarget
```

Currently available linear functionals:
```@docs
Empirikos.PriorDensity
Empirikos.MarginalDensity
```

Posterior estimands such as [`PosteriorMean`](@ref) can be typically decomposed into two linear functionals, a `numerator' and a `denominator':
```@docs
Base.numerator(::Empirikos.AbstractPosteriorTarget)
Base.denominator(::Empirikos.AbstractPosteriorTarget)
```