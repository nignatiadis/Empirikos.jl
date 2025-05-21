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
    mean_squares::T
    mean_squares_dof::S
    tstat::T
    # Inner constructor
    function NormalChiSquareSample(Z::T, S²::T, ν::S) where {T, S}
        tstat = Z / sqrt(S²)
        mean_squares_dof = ν + 1
        mean_squares = (ν + abs2(tstat)) * S² / mean_squares_dof

        # Create new instance with all fields
        new{T, S}(Z, S², ν, mean_squares, mean_squares_dof, tstat)
    end
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

function likelihood_distribution(Z::NormalChiSquareSample, μσ²)
    μ = μσ²[1]
    σ² = μσ²[2]
    dbn1 = Normal(μ, sqrt(σ²))
    dbn2 = likelihood_distribution(ScaledChiSquareSample(Z), σ²)
    product_distribution(dbn1, dbn2)
end

function likelihood_distribution(Z::NormalChiSquareSample, μσ²::NamedTuple{(:μ, :σ²)})
    μ = μσ².μ
    σ² = μσ².σ²
    dbn1 = Normal(μ, sqrt(σ²))
    dbn2 = likelihood_distribution(ScaledChiSquareSample(Z), σ²)
    product_distribution(dbn1, dbn2)
end

function likelihood_distribution(Z::NormalChiSquareSample, μσ²::NamedTuple{(:λ, :σ²)})
    σ² = μσ².σ²
    σ = sqrt(σ²)
    μ = μσ².λ * σ
    dbn1 = Normal(μ, σ)
    dbn2 = likelihood_distribution(ScaledChiSquareSample(Z), σ²)
    product_distribution(dbn1, dbn2)
end

function marginalize(Z::NormalChiSquareSample, prior::Distributions.ProductNamedTupleDistribution{(:λ, :σ²), Tuple{T, S}} where T<:Union{DiscreteNonParametric, Dirac} where S<:Union{DiscreteNonParametric,Dirac})
    prior_λ = prior.dists.λ
    prior_σ² = prior.dists.σ²

    
    λ_values = support(prior_λ)
    σ²_values = support(prior_σ²)
    λ_probs = probs(prior_λ)
    σ²_probs = probs(prior_σ²)
    
    total_components = length(λ_values) * length(σ²_values)
    
    # Calculate first component to determine concrete type
    first_params = (λ=first(λ_values), σ²=first(σ²_values))
    first_component = likelihood_distribution(Z, first_params)
    component_type = typeof(first_component)
    
    # Initialize properly typed arrays
    mixture_components = Vector{component_type}(undef, total_components)
    mixture_probs = Vector{Float64}(undef, total_components)
    
    # Generate all combinations
    idx = 1
    for (i, λ) in enumerate(λ_values)
        λ_prob = λ_probs[i]
        for (j, σ²) in enumerate(σ²_values)
            σ²_prob = σ²_probs[j]
            mixture_components[idx] = likelihood_distribution(Z, (λ=λ, σ²=σ²))
            mixture_probs[idx] = λ_prob * σ²_prob
            idx += 1
        end
    end
    
    return Distributions.MixtureModel(mixture_components, mixture_probs)
end

function marginalize(Z::NormalChiSquareSample, prior::Distributions.ProductNamedTupleDistribution{(:μ, :σ²), Tuple{T, S}} where T where S<:Dirac)
    prior_μ = prior.dists.μ
    prior_σ² = prior.dists.σ².value

    Z_normal = NormalSample(Z.Z, sqrt(prior_σ²))

    product_distribution(marginalize(Z_normal, prior_μ), 
        likelihood_distribution(ScaledChiSquareSample(Z), prior_σ²))
end

function marginalize(Z::NormalChiSquareSample, prior::Distributions.ProductNamedTupleDistribution{(:μ, :σ²), Tuple{T, S}} where T where S<:DiscreteNonParametric)
    prior_μ = prior.dists.μ
    prior_σ² = prior.dists.σ²

    σ²_values = support(prior_σ²)
    components = [marginalize(Z, product_distribution((μ=prior_μ, σ²=Dirac(σ²)))) for σ² in σ²_values]
    MixtureModel(components, probs(prior_σ²))
end


function marginalize(Z::NormalChiSquareSample, prior::Distributions.ProductNamedTupleDistribution{(:λ, :σ²), Tuple{T, S}} where T<:Dirac where S<:Dirac)
    likelihood_distribution(Z, (λ=prior.dists.λ.value, σ²=prior.dists.σ².value))  #distribution
end

function marginalize(Z::NormalChiSquareSample, prior::Distributions.ProductNamedTupleDistribution{(:λ, :σ²), Tuple{T, S}} where T where S<:Dirac)
    prior_λ = prior.dists.λ #distribution
    prior_σ² = prior.dists.σ².value #number
    prior_σ  = sqrt(prior_σ²)
    prior_μ = prior_σ * prior_λ 
    Z_normal = NormalSample(Z.Z, prior_σ)

    product_distribution(marginalize(Z_normal, prior_μ), 
        likelihood_distribution(ScaledChiSquareSample(Z), prior_σ²))
end

function marginalize(Z::NormalChiSquareSample, prior::Distributions.ProductNamedTupleDistribution{(:λ, :σ²), Tuple{T, S}} where T<:Normal where S<:DiscreteNonParametric)
    prior_λ = prior.dists.λ
    prior_σ² = prior.dists.σ²

    σ²_values = support(prior_σ²)
    components = [marginalize(Z, product_distribution((λ=prior_λ, σ²=Dirac(σ²)))) for σ² in σ²_values]
    MixtureModel(components, probs(prior_σ²))
end
