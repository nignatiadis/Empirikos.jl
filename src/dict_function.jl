Base.@kwdef struct DictFunction{S, C, T, D <: AbstractDict{S,T}}
    dict::D
    default_value::T = zero(eltype(values(dict)))
    discretizer::C = nothing
end

function DictFunction(dict::AbstractDict)
    DictFunction(dict, zero(eltype(values(dict))))
end

function DictFunction(keys, values; kwargs...)
    DictFunction(dict=SortedDict{eltype(keys), eltype(values)}(keys .=> values); kwargs...)
end


function (f::DictFunction{S})(x::S) where S
    get(f.dict, x, f.default_value)
end

function (f::DictFunction)(x)
    f(f.discretizer(x))
end

Base.keys(dictfun::DictFunction) = Base.keys(dictfun.dict)
Base.values(dictfun::DictFunction) = Base.values(dictfun.dict)
