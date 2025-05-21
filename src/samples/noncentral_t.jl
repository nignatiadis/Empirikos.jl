"""
    NoncentralTSample(Z, ν)

An observed sample ``Z`` drawn from a noncentral t-distribution with degrees of freedom ``\\nu`` and unknown noncentrality parameter ``\\lambda``.

```math
T \\sim t_{\\nu}(\\lambda)
```

``\\lambda`` is assumed unknown. The type above is used when the sample ``T`` is to be used for estimation or inference of ``\\lambda``.
"""
struct NoncentralTSample{T, S} <: ContinuousEBayesSample{T}
    Z::T
    ν::S
end

NoncentralTSample(ν) = NoncentralTSample(missing, ν)

function NoncentralTSample(Z::NormalChiSquareSample)
    NoncentralTSample(Z.tstat, Z.ν)
end 


eltype(::NoncentralTSample{T}) where {T} = T
support(::NoncentralTSample) = RealInterval(-Inf, +Inf)

response(Z::NoncentralTSample) = Z.Z
nuisance_parameter(Z::NoncentralTSample) = Z.ν
StatsBase.dof(Z::NoncentralTSample) = Z.ν



function likelihood_distribution(Z::NoncentralTSample, λ)
    ν = Z.ν
    NoncentralT(ν, λ)
end

function Base.show(io::IO, Z::NoncentralTSample)
    resp_Z = response(Z)
    print(io, "T(", resp_Z, "; λ, ν=", Z.ν,")")
end


function marginalize(Z::NoncentralTSample, prior::Normal)
    iszero(mean(prior)) || 
        error("Prior must be zero-centered for this marginalization method")
    v = sqrt(var(prior) + one(var(prior)))
    Distributions.AffineDistribution{typeof(v)}(zero(v), v, TDist(Z.ν))
end