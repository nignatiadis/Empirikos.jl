Base.@kwdef struct DictFunction{S, T, D <: AbstractDict{S,T}}
    dict::D
    default_value::T = zero(eltype(values(dict)))
end

function DictFunction(dict::AbstractDict)
    DictFunction(dict, zero(eltype(values(dict))))
end

function DictFunction(keys, values; kwargs...)
    DictFunction(dict=SortedDict{eltype(keys), eltype(values)}(keys .=> values); kwargs...)
end


function (f::DictFunction)(x)
    get(f.dict, x, f.default_value)
end

Base.keys(dictfun::DictFunction) = Base.keys(dictfun.dict)
Base.values(dictfun::DictFunction) = Base.values(dictfun.dict)


Base.@kwdef struct DiscretizedDictFunction{D <: Discretizer, F <: DictFunction}
    discretizer::D
    dictfunction::F
end

Base.keys(dictfun::DiscretizedDictFunction) = Base.keys(dictfun.dictfunction)

Base.values(dictfun::DiscretizedDictFunction) = Base.values(dictfun.dictfunction)

function DiscretizedDictFunction(discretizer, values; kwargs...)
    DiscretizedDictFunction(discretizer, DictFunction(Base.values(discretizer), values; kwargs...))
end

function (f::DiscretizedDictFunction)(x)
    f.dictfunction(f.discretizer(x))
end

function dictfun(discretizer::Discretizer, Zs::AbstractVector{<:EBayesSample}, f)
    skedasticity(Zs) == Homoskedastic() || error("Heteroskedastic likelihood not implemented.")
    Z = Zs[1]
    interval_Zs = [@set Z.Z = _int for _int in discretizer.sorted_intervals]
    if !isa(f, AbstractVector)
        f = f.(interval_Zs)
    end
    DiscretizedDictFunction(discretizer, DictFunction(interval_Zs, f))
end
