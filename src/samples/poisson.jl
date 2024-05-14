"""
    PoissonSample(Z, E)

An observed sample ``Z`` drawn from a Poisson distribution,

```math
Z \\sim \\text{Poisson}(\\mu \\cdot E).
```

The multiplying intensity ``E`` is assumed to be known (and equal to `1.0` by default), while
``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> PoissonSample(3)
𝒫ℴ𝒾(3; μ)
julia> PoissonSample(3, 1.5)
𝒫ℴ𝒾(3; μ⋅1.5)
```
"""
struct PoissonSample{T,S} <: DiscreteEBayesSample{T}
    Z::T
    E::S
end


PoissonSample(Z) = PoissonSample(Z, 1.0)
PoissonSample() = PoissonSample(missing)

response(Z::PoissonSample) = Z.Z
nuisance_parameter(Z::PoissonSample) = Z.E

likelihood_distribution(Z::PoissonSample, λ) = Poisson(λ * nuisance_parameter(Z))

summarize_by_default(Zs::Vector{<:PoissonSample}) = skedasticity(Zs) == Homoskedastic()

primary_parameter(::PoissonSample) = :μ


function Base.show(io::IO, Z::PoissonSample)
    resp_Z = response(Z)
    E = nuisance_parameter(Z)
    μ_string = E==1 ? "μ" : "μ⋅$(E)"
    print(io, "𝒫ℴ𝒾(", resp_Z,"; ",  μ_string,")")
end



# Conjugate computations

function default_target_computation(::BasicPosteriorTarget, ::PoissonSample, ::Gamma)
    Conjugate()
end

function marginalize(Z::PoissonSample, prior::Gamma)
    E = nuisance_parameter(Z)
    @unpack α, θ = prior
    β = 1 / θ
    p = β / (E + β)
    NegativeBinomial(α, p)
end

function posterior(Z::PoissonSample, prior::Gamma)
    E = nuisance_parameter(Z)
    @unpack α, θ = prior
    β = 1 / θ
    α_post = α + response(Z)
    β_post = β + E
    Gamma(α_post, 1 / β_post)
end





