struct Discretizer{S}
    sorted_intervals::S
end

Discretizer() = Discretizer(DataBasedDefault())

# Discretizer(intervals::S) where {T,C<:AbstractInterval{T},S<:AbstractVector{C}}


Base.keys(discr::Discretizer) = discr.sorted_intervals

function Discretizer(grid::AbstractVector; closed = :right, unbounded = :both)
    if (closed === :right) && (unbounded === :both)
        ints = EBInterval{eltype(grid)}[
            Interval{Unbounded,Closed}(nothing, grid[1])
            Interval{Open,Closed}.(grid[1:end-1], grid[2:end])
            Interval{Open,Unbounded}(grid[end], nothing)
        ]
    elseif (closed === :right) && (unbounded === :none)
        ints = Interval{Open,Closed}.(grid[1:end-1], grid[2:end])
    elseif (closed === :left) && (unbounded === :none)
        ints = Interval{Closed,Open}.(grid[1:end-1], grid[2:end])
    end
    Discretizer{typeof(ints)}(ints)
end

# TODO: What if x does not fall into any of the intervals?

function _discretize(sorted_intervals, x)
    n = length(sorted_intervals)
    left, right = 1, n

    for i = 1:n
        middle = div(left + right, 2)
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

# or maybe I should do sth else here?
# TODO: Check interval already exists in discretizer
function (discr::Discretizer)(Z::EBayesSample{<:Interval})
    Z
end

function broadcasted(discr::Discretizer, xs::AbstractVector{<:Number})
    C = eltype(discr.sorted_intervals)
    C[discr(x) for x in xs]
end

#----------------------------------------------------------------------------------------------
# TODO: These two should probably not be a convenience constructor and have a nice name instead
#----------------------------------------------------------------------------------------------
function default_discretizer(Zs::AbstractVector{<:EBayesSample}; eps = 1e-6, nbins = 300)
    _sample_min, _sample_max = extrema(response.(Zs))
    _grid = range(_sample_min - eps; stop = _sample_max + eps, length = nbins)
    Discretizer(_grid; unbounded = :none)
end

function default_discretizer(Zs::AbstractVector{<:Number}; eps = 1e-6, nbins = 300)
    _sample_min, _sample_max = extrema(Zs)
    _grid = range(_sample_min - eps; stop = _sample_max + eps, length = nbins)
    Discretizer(_grid; unbounded = :none)
end
#-------------------------------------------------------------------------------------------

function discretize(Zs::AbstractVector; kwargs...)
    discr = default_discretizer(Zs; kwargs...)
    discr.(Zs)
end

function summarize(Zs::AbstractVector, discr::Discretizer)
        summarize(discr.(Zs))
end

function summarize(Zs::AbstractVector, ws, discr::Discretizer)
    summarize(discr.(Zs), ws)
end

function _set_defaults(
    convexclass::Discretizer,
    Zs::AbstractVector;
    hints,
)
    eps = get(hints, :eps, 1e-6)
    nbins = get(hints, :nbins, 300)

    default_discretizer(Zs; eps=eps, nbins=nbins)
end
#function broadcasted(discr::Discretizer{T,C,S}, Zs::AbstractVector{<:EBayesSample}) where {T,C,S}
#    C[discr(x) for x in xs]
#end

# Why not SetDefault for this? uhm...

#function broadcasted(discr::Discretizer{T,C,S},
#                  xs::AbstractVector{<:EBayesSample}) where {T,C,S}
#   C[discr(x) for x in xs]
#end
