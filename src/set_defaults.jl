1+1

abstract type AbstractDefault end

struct DataBasedDefault <:  AbstractDefault end

function _set_defaults end

requires_defaults(::AbstractDefault) = true
requires_defaults(::Type) = false

function requires_defaults(obj)
    if nfields(obj) == 0
        return false
    else
        _fields = getfield.(Ref(obj), fieldnames(typeof(obj)))
         any(requires_defaults.(_fields))
    end
end


function set_defaults(obj, data...; hints = Dict())
    hints = Dict{Any,Any}(hints)
    get!(hints, :field_parents, [])

    if !requires_defaults(obj)
        return obj
    end

    _fieldnames = fieldnames(typeof(obj))
    _fields = getfield.(Ref(obj), _fieldnames)

    if any(isa.(_fields, AbstractDefault ))
        return _set_defaults(obj, data...; hints=hints)
    end

    idx = findfirst(requires_defaults.(_fields))
    subfield_hints = deepcopy(hints)
    push!(subfield_hints[:field_parents], obj)
    subfield = set_defaults(_fields[idx], data...; hints=subfield_hints)

    _lens = Setfield.PropertyLens{_fieldnames[idx]}()
    set_defaults(set(obj, _lens, subfield), data...; hints=hints)
end
