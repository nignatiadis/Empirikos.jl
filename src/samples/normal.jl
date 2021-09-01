abstract type AbstractNormalSample{T} <: ContinuousEBayesSample{T} end

"""
    NormalSample(Z,σ)

An observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z \\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> NormalSample(0.5, 1.0)          #Z=0.5, σ=1
Z=     0.5 | σ=1.0
```
"""
struct NormalSample{T,S} <: AbstractNormalSample{T}
    Z::T
    σ::S
end

# TODO: Should not need this eventually.
function NormalSample(Z::P, σ::S) where {T,P<:EBInterval{T},S}
    NormalSample{EBInterval{T},S}(Z, σ)
end

function NormalSample(σ::S) where {S}
    NormalSample(missing, σ)
end



"""
    StandardNormalSample(Z)

An observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 =1``.

```math
Z \\sim \\mathcal{N}(\\mu, 1)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```jldoctest
julia> StandardNormalSample(0.5)          #Z=0.5
Z=     0.5 | σ=1.0
```
"""
struct StandardNormalSample{T} <: AbstractNormalSample{T}
    Z::T
end

StandardNormalSample() = StandardNormalSample(missing)

eltype(Z::AbstractNormalSample{T}) where {T} = T
support(Z::AbstractNormalSample) = RealInterval(-Inf, +Inf)

response(Z::AbstractNormalSample) = Z.Z
var(Z::AbstractNormalSample) = Z.σ^2
var(Z::StandardNormalSample) = one(eltype(response(Z)))
var(Z::StandardNormalSample{Missing}) = 1.0

std(Z::AbstractNormalSample) = Z.σ
std(Z::StandardNormalSample) = one(eltype(response(Z)))
std(Z::StandardNormalSample{Missing}) = 1.0

nuisance_parameter(Z::AbstractNormalSample) = std(Z)

likelihood_distribution(Z::AbstractNormalSample, μ) = Normal(μ, std(Z))


function Base.show(io::IO, Z::AbstractNormalSample{<:Real})
    Zz = response(Z)
    print(io, "Z=")
    print(io, lpad(round(Zz, digits = 4),8))
    print(io, " | ", "σ=")
    print(io, rpad(round(std(Z), digits = 3),5))
end


function Base.show(io::IO, Z::AbstractNormalSample{<:Interval})
    Zz = response(Z)
    print(io, "Z ∈ ")
    show(IOContext(io, :compact => true), Zz)
    print(io, " | ", "σ=")
    print(io, rpad(round(std(Z), digits = 3),5))
end



# Targets

# TODO: Note this is not correct for intervals.
function cf(target::MarginalDensity{<:AbstractNormalSample}, t)
    error_dbn = likelihood_distribution(location(target))
    cf(error_dbn, t)
end


# Conjugate computations
function default_target_computation(::BasicPosteriorTarget,
    ::AbstractNormalSample,
    ::Normal
)
    Conjugate()
end

function marginalize(Z::AbstractNormalSample, prior::Normal)
    prior_var = var(prior)
    prior_μ = mean(prior)
    likelihood_var = var(Z)
    marginal_σ = sqrt(likelihood_var + prior_var)
    Normal(prior_μ, marginal_σ)
end


function posterior(Z::AbstractNormalSample, prior::Normal)
    z = response(Z)
    sigma_squared = var(Z)
    prior_mu = mean(prior)
    prior_A = var(prior)

    post_mean =
        (prior_A) / (prior_A + sigma_squared) * z +
        sigma_squared / (prior_A + sigma_squared) * prior_mu
    post_var = prior_A * sigma_squared / (prior_A + sigma_squared)
    Normal(post_mean, sqrt(post_var))
end




function _set_defaults(
    convexclass::DiscretePriorClass,
    Zs::VectorOrSummary{<:AbstractNormalSample};  #TODO for MultinomialSummary
    hints,
)
    eps = get(hints, :eps, 1e-4)
    prior_grid_length = get(hints, :prior_grid_length, 300)::Integer
    _sample_min, _sample_max = extrema(response.(Zs))
    # This won't handle infinity correctly. TODOO
    # Also TODO once switch from Intervals.jl to IntervalSets.jl
    _sample_min = isa(_sample_min, Interval) ? first(_sample_min) : _sample_min
    _sample_max = isa(_sample_max, Interval) ? last(_sample_max) : _sample_max
    _grid = range(_sample_min - eps; stop = _sample_max + eps, length = prior_grid_length)
    DiscretePriorClass(_grid)
end


function _set_defaults(
    convexclass::GaussianScaleMixtureClass,
    Zs::AbstractVector{<:AbstractNormalSample};  #TODO for MultinomialSummary
    hints,
)
    grid_scaling = get(hints, :grid_scaling, sqrt(2))

    σ_min =  minimum(std.(Zs))./ 10
    _max = maximum(response.(Zs).^2 .-  var.(Zs))
    _σ_max = _max > 0.0 ? 2*sqrt(_max) : 8*σ_min
    σ_max = get(hints, :σ_max, _σ_max) #somewhat redundant computations above.

    npoint = ceil(Int, log2(σ_max/σ_min)/log2(grid_scaling))
    σ_grid = σ_min*grid_scaling.^(0:npoint)

    GaussianScaleMixtureClass(σ_grid)
end

# Target specifics
function Base.extrema(density::MarginalDensity{<:AbstractNormalSample{<:Real}})
    (0.0, 1 / sqrt(2π * var(location(density))))
end
