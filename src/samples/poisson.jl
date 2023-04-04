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
Z=3  | E=1.0
julia> PoissonSample(3, 1.5)
Z=3  | E=1.5
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



# DiscretePriorClass


function _set_defaults(
    convexclass::DiscretePriorClass,
    Zs::VectorOrSummary{<:PoissonSample};
    hints,
)
    eps = get(hints, :eps, 1e-4)
    prior_grid_length = get(hints, :prior_grid_length, 300)::Integer
    _sample_min, _sample_max = extrema(response.(Zs) ./ nuisance_parameter.(Zs))
    _grid_min = max(2 * eps, _sample_min - eps)
    _grid_max = _sample_max + eps
    DiscretePriorClass(range(_grid_min; stop = _grid_max, length = prior_grid_length))
end
