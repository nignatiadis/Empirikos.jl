function autoconvexclass(𝒢, Zs; kwargs...)
    autoconvexclass(𝒢; kwargs...)
end 

autoconvexclass(𝒢; kwargs...) = 𝒢

#--------------------------------------------------
# DiscretePriorClass 
#--------------------------------------------------
function autoconvexclass(
    ::DiscretePriorClass{Nothing},
    Zs::VectorOrSummary{<:AbstractNormalSample};  #TODO for MultinomialSummary
    eps = 1e-4,
    prior_grid_size = 300
)
    _sample_min, _sample_max = extrema(response.(Zs))

    #_sample_min = isa(_sample_min, Interval) ? first(_sample_min) : _sample_min
    #_sample_max = isa(_sample_max, Interval) ? last(_sample_max) : _sample_max

    _grid = range(_sample_min - eps; stop = _sample_max + eps, length = prior_grid_size)
    DiscretePriorClass(_grid)
end


function autoconvexclass(
    ::DiscretePriorClass{Nothing},
    Zs::VectorOrSummary{<:PoissonSample};
    eps = 1e-4,
    prior_grid_size = 300
)
    _sample_min, _sample_max = extrema(response.(Zs) ./ nuisance_parameter.(Zs))
    _grid_min = max(2 * eps, _sample_min - eps)
    _grid_max = _sample_max + eps
    DiscretePriorClass(range(_grid_min; stop = _grid_max, length = prior_grid_size))
end

function autoconvexclass(
    ::DiscretePriorClass{Nothing},
    Zs::VectorOrSummary{<:BinomialSample};
    eps=1e-4,
    prior_grid_size = 300
)
    DiscretePriorClass(range(eps; stop = 1 - eps, length = prior_grid_size))
end


function autoconvexclass(
    ::DiscretePriorClass{Nothing},
    Ss::AbstractVector{<:ScaledChiSquareSample};
    prior_grid_size = 300,
    lower_quantile = 0.01,
)

    a_min = quantile(response.(Ss), lower_quantile)
    a_max = maximum(response.(Ss))

    grid = exp.(range(start = log(a_min), stop = log(a_max), length = prior_grid_size))
    _prior = DiscretePriorClass(grid)
    _prior
end


#--------------------------------------------------
# GaussianScaleMixtureClass 
#--------------------------------------------------
function autoconvexclass(::GaussianScaleMixtureClass{Nothing};
    σ_min, σ_max, grid_scaling = sqrt(2))
    npoint = ceil(Int, log2(σ_max/σ_min)/log2(grid_scaling))
    σ_grid = σ_min*grid_scaling.^(0:npoint)
    GaussianScaleMixtureClass(σ_grid)
end


function autoconvexclass(
    𝒢::GaussianScaleMixtureClass{Nothing},
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

#--------------------------------------------------
# BetaMixtureClass 
#--------------------------------------------------

function autoconvexclass(::BetaMixtureClass{Nothing}; bandwidth = 0.05, grid = 0:0.01:1)
    αs = 1 .+ (grid ./bandwidth)
    βs = 1 .+ ((1 .- grid) ./bandwidth)
    BetaMixtureClass(αs, βs)
end




