#---------------------------------------
# types for a single EB sample
#---------------------------------------
abstract type EBayesSample{T} end

function likelihood_distribution end 
likelihood(Z::EBayesSample, param) = pdf(likelihood_distribution(Z, param), response(Z))
loglikelihood(Z::EBayesSample, param) = logpdf(likelihood_distribution(Z, param), response(Z))


# basic interface 
# pdf(Z, NormalSample()) -> marginalization ...
# posterior(sample , prior) -> Distribution...  
# marginal()

# target(Z)(prior) -> 
# target(Z)(μ) = μ # dirac mass prior type of computation 

abstract type AbstractNormalSample{T} <: EBayesSample{T} end

for NS in [:NormalSample, :HomoskedasticNormalSample]
	@eval begin
		struct $NS{T,S} <: AbstractNormalSample{T}
			Z::T
 	        σ::S
		end
		
		function $NS(Z::T) where {T}
		    $NS(Z, one(T))
		end
	end
end

@doc """
    NormalSample(Z,σ)
    HomoskedasticNormalSample(Z,σ)

A observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 > 0``.

```math
Z \\sim \\mathcal{N}(\\mu, \\sigma^2)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.
The two types behave identically on scalars, however they allow for distinct dispatch when working with
`AbstractArray{T} where T<:NormalSample`, resp. `T<:HomoskedasticNormalSample`.

```julia
NormalSample(0.5, 1.0)          #Z=0.5, σ=1
HomoskedasticNormalSample(0.5, 1.0)          #Z=0.5, σ=1
```
""" NormalSample, HomoskedasticNormalSample


"""
    StandardNormalSample(Z)

A observed sample ``Z`` drawn from a Normal distribution with known variance ``\\sigma^2 =1``.

```math
Z \\sim \\mathcal{N}(\\mu, 1)
```

``\\mu`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\mu``.

```julia
StandardNormalSample(0.5)          #Z=0.5
```
"""
struct StandardNormalSample{T} <: AbstractNormalSample{T}
    Z::T
end


eltype(Z::AbstractNormalSample{T}) where {T} = T
zero(Z::AbstractNormalSample{T}) where {T} = zero(T)  #debatable?
support(Z::AbstractNormalSample) = RealInterval(-Inf, +Inf)


response(Z::AbstractNormalSample) = Z.Z
var(Z::AbstractNormalSample) = Z.σ^2
var(Z::StandardNormalSample) = one(eltype(Z))

std(Z::AbstractNormalSample) = Z.σ
std(Z::StandardNormalSample) = one(eltype(Z))

	
#zeos(ss::AbstractNormalSamples) = zeros(eltype(response(ss)), length(ss))

likelihood_distribution(Z::AbstractNormalSample, μ) = Normal(μ, std(Z))


# 






# Poisson

# Binomial

# Replicated , ReplicatedArray