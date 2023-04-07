"""
    NormalChiSquareSample(Z, S², ν)


The sample is assumed to be
An observed sample ``Z`` drawn from a scaled chi-square distribution with unknown scale ``\\sigma^2 > 0``.

```math
Z \\sim \\frac{\\sigma^2}{\\nu}}\\chi^2_{\\nu}
```

``\\sigma^2`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.
"""
struct NormalChiSquareSample{T, S} <: EBayesSample{T}
    Z::T
    S²::T
    ν::S
end

function response(Z::NormalChiSquareSample)
    [Z.Z, Z.S²]
 end

 function nuisance_parameter(Z::NormalChiSquareSample)
    Z.ν
 end

# convert

function ScaledChiSquareSample(Z::NormalChiSquareSample)
    ScaledChiSquareSample(Z.S², Z.ν)
end

function Base.show(io::IO, Z::NormalChiSquareSample)
    z, s² = response(Z)
    ν = Z.ν
    print(io,  "𝒩(", z, ";μ,σ)", "⊗",  "ScaledΧ²(", s², ";σ²,ν=", ν,")")
end

function Empirikos.likelihood_distribution(Z::NormalChiSquareSample, μσ²)
    μ = μσ²[1]
    σ² = μσ²[2]
    dbn1 = Normal(μ, σ²)
    dbn2 = likelihood_distribution(ScaledChiSquareSample(Z), σ²)
    product_distribution(dbn1, dbn2)
end
