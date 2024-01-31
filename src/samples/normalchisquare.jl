



"""
    NormalChiSquareSample(Z, S², ν)

This type represents a tuple ``(Z, S^2)`` consisting of the following two measurements:
* `Z`, a Gaussian measurement ``Z \\sim \\mathcal{N}(\\mu, \\sigma^2)`` centered around ``\\mu`` with variance ``\\sigma^2``,
* `S²`, an independent unbiased measurement ``S^2`` of ``\\sigma^2`` whose law is the scaled ``\\chi^2`` distribution with `ν` (``\\nu \\geq 1``) degrees of freedom:

```math
(Z, S) \\, \\sim \\, \\mathcal{N}(\\mu, \\sigma^2) \\otimes \\frac{\\sigma^2}{\\nu} \\chi^2_{\\nu}.
```

Here ``\\sigma^2 > 0`` and ``\\mu \\in \\mathbb R`` are assumed unknown.
``(Z, S^2)`` is to be used for estimation or inference of ``\\mu`` and ``\\sigma^2``.
"""
struct NormalChiSquareSample{T, S} <: EBayesSample{T}
    Z::T
    S²::T
    ν::S
end

function NormalChiSquareSample(Z, S²::ScaledChiSquareSample)
    NormalChiSquareSample(Z, response(S²), S².ν)
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
