struct CompoundSample{T, EBS, S} <: EBayesSample{T}
    vec::AbstractVector{EBS}
    probs::AbstractVector{S}
    Z::T
end

response(Z::CompoundSample) = Z.Z
nuisance_parameter(Z::CompoundSample) = (Z.vec, Z.probs)


function Base.isless(a::CompoundSample, b::CompoundSample)
    nuisance_parameter(a) == nuisance_parameter(b) ||
        throw(ArgumentError("Comparison only implemented for homoskedastic compound samples"))
    Base.isless(response(a), response(b))
end


function compound(Zs::AbstractVector{<:EBayesSample})
    n = nobs(Zs)
    Zs_no_data = set_response.(Zs)
    Zs_unique_dict = countmap(Zs_no_data)
    Zs_unique = collect(keys(Zs_unique_dict))
    Zs_unique_prob = values(Zs_unique_dict)./n
    CompoundSample.(Ref(Zs_unique), Ref(Zs_unique_prob), response.(Zs))
end

function compound(Zs::MultinomialSummary{<:EBayesSample})
    n = nobs(Zs)
    Zs_no_data = set_response.(collect(keys(Zs)))
    _mult = fweights(multiplicity(Zs))
    Zs_unique_dict = countmap(Zs_no_data, _mult)
    Zs_unique = collect(keys(Zs_unique_dict))
    Zs_unique_prob = collect(values(Zs_unique_dict))./n
    Zs_compound = CompoundSample.(Ref(Zs_unique), Ref(Zs_unique_prob), response.(collect(keys(Zs))))
    summarize(Zs_compound, _mult)
end


function likelihood_distribution(Z::CompoundSample, μ)
    n = length(Z.vec)
    MixtureModel(likelihood_distribution.(Z.vec, μ), Z.probs)
end

function marginalize(Z::CompoundSample, prior::Distribution)
    n = length(Z.vec)
    MixtureModel(marginalize.(Z.vec, Ref(prior)),  Z.probs)
end

# TODO: # beware a bit of this, hopefully compound won't be used too much
function skedasticity(Zs::AbstractArray{<:CompoundSample,1})
    Empirikos.Homoskedastic()
end

# hard code as missing.
struct HeteroskedasticSamples{EBS <: EBayesSample{Missing}, S}
    vec::AbstractVector{EBS}
    probs::AbstractVector{S}
end

function heteroskedastic(Zs::AbstractVector{<:EBayesSample})
    _comp = compound(Zs)[1]
    HeteroskedasticSamples( _comp.vec, _comp.probs)
end

function heteroskedastic(Zs::MultinomialSummary{<:EBayesSample})
    _comp = collect(keys(compound(Zs)))[1]
    HeteroskedasticSamples( _comp.vec, _comp.probs)
end

function CompoundSample(Zs::HeteroskedasticSamples)
    CompoundSample(Zs.vec, Zs.probs, missing)
end
