"""
    ConvexPriorClass

Abstract type representing convex classes of probability distributions ``\\mathcal{G}``.
"""
abstract type ConvexPriorClass end


struct PriorVariable{C<:ConvexPriorClass,V}
    convexclass::C
    finite_param::V
    model::Any
end


Base.broadcastable(prior::PriorVariable) = Ref(prior)

function (priorvariable::PriorVariable)(p::AbstractVector{<:Real})
    convexclass = priorvariable.convexclass
    convexclass(p)
end

abstract type AbstractMixturePriorClass <: ConvexPriorClass end

# TODO: implement correct projection onto simplex and check deviation is not too big
# though this should mostly help with minor numerical difficulties
function fix_πs(πs)
    πs = max.(πs, 0.0)
    πs = πs ./ sum(πs)
end

"""
    DiscretePriorClass

"""
struct DiscretePriorClass{S} <: AbstractMixturePriorClass
    support::S #(-Inf, Inf) #default
end

DiscretePriorClass() = DiscretePriorClass(nothing)

function (convexclass::DiscretePriorClass)(p::AbstractVector{<:Real})
    DiscreteNonParametric(support(convexclass), fix_πs(p))
end

support(convexclass::DiscretePriorClass) = convexclass.support
nparams(convexclass::DiscretePriorClass) = length(support(convexclass))

function prior_variable!(model, convexclass::AbstractMixturePriorClass; var_name = "π") # adds constraints
    n = nparams(convexclass)
    tmp_vars = @variable(model, [i = 1:n])
    #model[Symbol(var_name)] = tmp_vars
    set_lower_bound.(tmp_vars, 0.0)
    #@constraint(model, tmp_vars .> 0)
    con = @constraint(model, sum(tmp_vars) == 1.0)
    PriorVariable(convexclass, tmp_vars, model)
end

# Dirac
function pdf(prior::PriorVariable{<:DiscretePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    pdf_combination = likelihood.(Z, support(convexclass))
    @expression(model, dot(finite_param, pdf_combination))
end

# Dirac
function cdf(prior::PriorVariable{<:DiscretePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    cdf_combination = cdf.(likelihood_distribution.(Z, support(convexclass)), response(Z))
    @expression(model, dot(finite_param, cdf_combination))
end

function (target::LinearEBayesTarget)(prior::PriorVariable{<:DiscretePriorClass})
    @unpack convexclass, finite_param, model = prior
    linear_functional_evals = target.(support(convexclass))
    @expression(model, dot(finite_param, linear_functional_evals))
end



"""
    MixturePriorClass

"""
struct MixturePriorClass{S} <: AbstractMixturePriorClass
    components::S
end

MixturePriorClass() = MixturePriorClass(nothing)
components(mixclass::MixturePriorClass) = mixclass.components
nparams(mixclass::MixturePriorClass) = length(components(mixclass))

function (mixclass::MixturePriorClass)(p::AbstractVector{<:Real})
    MixtureModel(components(mixclass), fix_πs(p))
end

function pdf(prior::PriorVariable{<:MixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    pdf_combination = pdf.(components(convexclass), Z)
    @expression(model, dot(finite_param, pdf_combination))
end

function cdf(prior::PriorVariable{<:MixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    cdf_combination = cdf.(components(convexclass), Z)
    @expression(model, dot(finite_param, cdf_combination))
end

function (target::LinearEBayesTarget)(prior::PriorVariable{<:MixturePriorClass})
    @unpack convexclass, finite_param, model = prior
    linear_functional_evals = target.(components(convexclass))
    @expression(model, dot(finite_param, linear_functional_evals))
end
