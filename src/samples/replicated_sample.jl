struct DependentReplicatedSample{T, S<:AbstractVector{T}} <: Empirikos.EBayesSample{Float64}
    Zs::S
end

function flatten_and_weight(samples::Vector{<:DependentReplicatedSample})
    # Concatenate all Zs fields
    flattened = vcat(getproperty.(samples, :Zs)...)
    
    # Create a weight vector
    weights_vec = Float64[]
    for sample in samples
        weight = length(sample.Zs)
        append!(weights_vec, fill(1/weight, weight))
    end

    # Normalize weights so they sum up to 1
    return flattened, weights(weights_vec)
end

function summarize(Zs::AbstractVector{<:DependentReplicatedSample}; flatten::Bool=true)
    flattened_Zs, flattened_weights = flatten_and_weight(Zs)
    Empirikos.MultinomialSummary(SortedDict(countmap(flattened_Zs, flattened_weights)); effective_nobs = length(Zs))
end