struct CompoundSample{T, EBS} <: EBayesSample{T}
    vec::AbstractVector{EBS}
    Z::T
end

function compound(Zs::AbstractVector{<:EBayesSample})
    CompoundSample.(Ref(Zs), response.(Zs))
end

Empirikos.response(Z::CompoundSample) = Z.Z

function Empirikos.likelihood_distribution(Z::CompoundSample, μ)
    n = length(Z.vec)
    MixtureModel(likelihood_distribution.(Z.vec, μ), fill(1/n, n))
end

function Empirikos.marginalize(Z::CompoundSample, prior::Distribution)
    n = length(Z.vec)
    MixtureModel(marginalize.(Z.vec, Ref(prior)),  fill(1/n, n))
end

function Empirikos.skedasticity(Zs::AbstractArray{<:CompoundSample,1})
    Empirikos.Heteroskedastic()
end
