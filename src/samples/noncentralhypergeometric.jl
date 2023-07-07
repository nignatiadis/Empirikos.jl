"""
    NonCentralHypergeometricSample

Empirical Bayes sample type used to represent a 2×2 contigency table drawn from
Fisher's noncentral hypergeometric distribution conditionally on table margins.
The goal is to conduct inference for the log odds ratio θ.

More concretely, suppose we observe the following contigency table.
|         | Outcome 1 | Outcome 2 |         |
|:-------:|:---------:|:---------:|:-------:|
|Stratum 1| Z₁        | X₁        | n₁      |
|Stratum 2| Z₂        | X₂        | n₂      |
|         | Z₁pZ₂     | .         | .       |

This table can be turned into an empirical Bayes sample through either of the following calls:

```julia
NonCentralHypergeometricSample(Z₁, n₁, n₂, Z₁pZ₂)
NonCentralHypergeometricSample(Z₁, X₁, Z₂, X₂; margin_entries=false)
```

The likelihood of the above as a function of the log odds ratio θ is given by:
```math
\\frac{\\binom{n_1}{Z_1}\\binom{n_2}{Z_2} \\exp(\\theta Z_1)}{\\sum_{t}\\binom{n_1}{t}\\binom{n_2}{Z_1pZ_2 - t}\\exp(\\theta t)}.
```

"""
struct NonCentralHypergeometricSample{T} <: DiscreteEBayesSample{T}
    Z₁::T
    X₁::T
    Z₂::T
    X₂::T
    n₁::T
    n₂::T
    Z₁pZ₂::T
end

function NonCentralHypergeometricSample(a,b,c,d; margin_entries=true)
    Z₁ = a
    if margin_entries
        n₁ = b
        n₂ = c
        Z₁pZ₂ = d
        X₁ = n₁ - Z₁
        Z₂ = Z₁pZ₂ - Z₁
        X₂ = n₂ - Z₂
    else
        X₁ = b
        Z₂ = c
        X₂ = d
        n₁ = Z₁ + X₁
        n₂ = Z₂ + X₂
        Z₁pZ₂ = Z₁ + Z₂
    end
    NonCentralHypergeometricSample(Z₁,X₁,Z₂,X₂,n₁,n₂,Z₁pZ₂)
end


function Base.show(io::IO, Z::NonCentralHypergeometricSample)
    print(io, "HGeom({", Z.Z₁,"/", Z.X₁,"}/{", Z.Z₂, "/", Z.Z₂,"}; θ)")
end


function response(Z::NonCentralHypergeometricSample)
    Z.Z₁
 end

function likelihood_distribution(Z::NonCentralHypergeometricSample, θ)
    FisherNoncentralHypergeometric(Z.n₁, Z.n₂, Z.Z₁pZ₂, exp(θ))
end


function odds_ratio(Z::NonCentralHypergeometricSample; offset=0.5)
    num = (Z.Z₁ + offset)/(Z.X₁ + offset)
    denom = (Z.Z₂ + offset)/(Z.X₂ + offset)
    num/denom
end
