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

#--------------------------------------------------
# UniformScaleMixtureClass 
#--------------------------------------------------
function autoconvexclass(::UniformScaleMixtureClass{Nothing};
    a_min, a_max, grid_scaling=√2)
    npoint = ceil(Int, log(a_max/a_min)/log(grid_scaling))
    a_grid = a_min .* grid_scaling .^ (0:npoint)
    UniformScaleMixtureClass(a_grid)
end

function autoconvexclass(
    𝒢::UniformScaleMixtureClass{Nothing},
    Zs::AbstractVector{<:AbstractNormalSample};
    a_min=nothing, a_max=nothing, kwargs...
)
    if isnothing(a_min)
        a_min = (minimum(std.(Zs)) / 10) * √3 
    end
    
    if isnothing(a_max)
        m    = maximum(response.(Zs).^2 .-  var.(Zs))
        a_max = m > 0 ? 2 * sqrt(3*m) : (8*a_min)
    end
    
    autoconvexclass(𝒢; a_min=a_min, a_max=a_max, kwargs...)
end


#--------------------------------------------------
# GaussianLocationScaleMixtureClass 
#--------------------------------------------------
function autoconvexclass(::GaussianLocationScaleMixtureClass{Nothing};
    μ_min, μ_max, std, σ_min, σ_max, grid_scaling=√2)
    step_μ = std / 4
    μ_grid = μ_min:step_μ:μ_max
    npoint_σ = ceil(Int, log2(σ_max/σ_min)/log2(grid_scaling))
    σ_grid = σ_min*grid_scaling.^(0:npoint_σ)
    
    GaussianLocationScaleMixtureClass(μ_grid, std, σ_grid)
end

function autoconvexclass(
    𝒢::GaussianLocationScaleMixtureClass{Nothing},
    Zs::AbstractVector{<:AbstractNormalSample};
    μ_min=nothing, μ_max=nothing, σ_min=nothing, σ_max=nothing, kwargs...
)
       #std = (minimum(std.(Zs)) / 10)
    #end
    if isnothing(μ_min)
        μ_min = 0.005
    end

    if isnothing(μ_max)
        μ_max = 6
    end

    if isnothing(σ_min)
        σ_min = minimum(std.(Zs))./ 10
    end

    if isnothing(σ_max)
        _max = maximum(response.(Zs).^2 .-  var.(Zs))
        σ_max =  _max > 0.0 ? 2*sqrt(_max) : 8*σ_min
    end
    
    autoconvexclass(𝒢; μ_min=μ_min, μ_max=μ_max, σ_min=σ_min, σ_max=σ_max, kwargs...)
end


