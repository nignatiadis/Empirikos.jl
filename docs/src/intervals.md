# Confidence intervals

Here we describe two methods for forming confidence intervals for empirical Bayes estimands ([`Empirikos.EBayesTarget`](@ref)).

## F-Localization based intervals

```@docs
FLocalizationInterval
```

## AMARI intervals
```@docs
AMARI
```

## Interface

```@docs
confint(::AMARI, ::Empirikos.AbstractPosteriorTarget, ::Any)
```