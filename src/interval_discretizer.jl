struct Discretizer{T, C <: AbstractInterval{T}, S<:AbstractVector{C}}
    sorted_intervals::S
end

Base.keys(discr::Discretizer) = discr.sorted_intervals

function Discretizer(grid::AbstractVector; closed=:right, unbounded=:both)
    if (closed === :right) && (unbounded === :both)
        ints = EBInterval{eltype(grid)}[Interval{Unbounded, Closed}(nothing, grid[1]);
                                    Interval{Open, Closed}.(grid[1:end-1], grid[2:end]);
                                    Interval{Open, Unbounded}(grid[end], nothing)]
    elseif (closed === :right) && (unbounded === :none)
        ints = Interval{Open, Closed}.(grid[1:end-1], grid[2:end])
    end
    Discretizer(ints)
end


function _discretize(sorted_intervals, x)
    n = length(sorted_intervals)
    left, right = 1, n

    for i=1:n
        middle = div(left+right, 2)
        middle_interval = sorted_intervals[middle]
        if x ∈ middle_interval
            return middle_interval
        elseif isless(x, Intervals.RightEndpoint(middle_interval))
            right = middle - 1
        else
            left = middle + 1
        end
    end
    middle_interval
end

function (discr::Discretizer)(x)
    _discretize(discr.sorted_intervals, x)
end

function (discr::Discretizer)(Z::EBayesSample)
    # define response! instead?
    @set Z.Z = discr(Z.Z)
end

function broadcasted(discr::Discretizer{T,C,S}, xs::AbstractVector{<:Number}) where {T,C,S}
    C[discr(x) for x in xs]
end


function Discretizer(Zs::AbstractVector{<:EBayesSample}; eps=1e-6, nbins=300, kwargs...)
    _sample_min, _sample_max = extrema(response.(Zs))
    _grid = range(_sample_min - eps; stop=_sample_max + eps, length=nbins)
    Discretizer(_grid; kwargs...)
end


function discretize(Zs::AbstractVector; kwargs...)
    discr = Discretizer(Zs; kwargs...)
    discr.(Zs)
end



#function broadcasted(discr::Discretizer{T,C,S}, Zs::AbstractVector{<:EBayesSample}) where {T,C,S}
#    C[discr(x) for x in xs]
#end


#function broadcasted(discr::Discretizer{T,C,S},
  #                  xs::AbstractVector{<:EBayesSample}) where {T,C,S}
 #   C[discr(x) for x in xs]
#end
