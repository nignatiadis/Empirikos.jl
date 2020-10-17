struct DictFunction{T,S,D <: AbstractDict{S,T}}
    dict::D
    default_value::T
end

function DictFunction(dict)
    DictFunction(dict, zero(eltype(values(dict))))
end

function (f::DictFunction)(x)
    get(f.dict, x, f.default_value)
end
