function autoconvexclass(::GaussianScaleMixtureClass;
    σ_min, σ_max, grid_scaling = sqrt(2))
    npoint = ceil(Int, log2(σ_max/σ_min)/log2(grid_scaling))
    σ_grid = σ_min*grid_scaling.^(0:npoint)
    GaussianScaleMixtureClass(σ_grid)
end

function autoconvexclass(
    𝒢::GaussianScaleMixtureClass,
    Zs::AbstractVector{<:AbstractNormalSample};  #TODO for MultinomialSummary
    σ_min = nothing, σ_max = nothing, kwargs...)

    if isnothing(σ_min)
        σ_min = minimum(std.(Zs))./ 10
    end

    if isnothing(σ_max)
        _max = maximum(response.(Zs).^2 .-  var.(Zs))
        σ_max =  _max > 0.0 ? 2*sqrt(_max) : 8*σ_min
    end

    autoconvexclass(𝒢; σ_min=σ_min, σ_max=σ_max, kwargs...)
end
