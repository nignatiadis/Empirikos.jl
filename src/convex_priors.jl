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

function (convexclass::ConvexPriorClass)(p)
    PriorVariable(convexclass, v)
end

struct DiscretePriorClass{S} <: ConvexPriorClass
    support::S #(-Inf, Inf) #default
end

function (convexclass::DiscretePriorClass)(p::AbstractVector{<:Real})
    DiscreteNonParametric(support(convexclass), p)
end

support(convexclass::DiscretePriorClass) = convexclass.support
nparams(convexclass::DiscretePriorClass) = length(support(convexclass))

function prior_variable!(model, convexclass::DiscretePriorClass; var_name = "π") # adds constraints
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

function linear_functional end
