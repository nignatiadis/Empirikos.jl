"""
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

function (priorvariable::PriorVariable)()
    priorvariable(JuMP.value.(priorvariable.finite_param))
end

abstract type AbstractSimplexPriorClass <: ConvexPriorClass end
abstract type AbstractMixturePriorClass <: AbstractSimplexPriorClass end

# TODO: implement correct projection onto simplex and check deviation is not too big
# though this should mostly help with minor numerical difficulties
function fix_πs(πs)
    πs = max.(πs, 0.0)
    πs = πs ./ sum(πs)
end

"""
    DiscretePriorClass(support) <: Empirikos.ConvexPriorClass

Type representing the family of all discrete distributions supported on a subset
of `support`, i.e., it represents all `DiscreteNonParametric` distributions with
`support = support` and `probs` taking values on the probability simplex.

Note that `DiscretePriorClass(support)(probs) == DiscreteNonParametric(support, probs)`.

# Examples
```julia-repl
julia> gcal = DiscretePriorClass([0,0.5,1.0])
DiscretePriorClass | suport = [0.0, 0.5, 1.0]
julia> gcal([0.2,0.2,0.6])
DiscreteNonParametric{Float64,Float64,Array{Float64,1},Array{Float64,1}}(support=[0.0, 0.5, 1.0], p=[0.2, 0.2, 0.6])
```
"""
struct DiscretePriorClass{S} <: AbstractSimplexPriorClass
    support::S
end

DiscretePriorClass(; support=DataBasedDefault()) = DiscretePriorClass(support)

function (convexclass::DiscretePriorClass)(p::AbstractVector{<:Real})
    DiscreteNonParametric(support(convexclass), fix_πs(p))
end

support(convexclass::DiscretePriorClass) = convexclass.support
nparams(convexclass::DiscretePriorClass) = length(support(convexclass))

function Base.show(io::IO, gcal::DiscretePriorClass)
    print(io, "DiscretePriorClass | suport = ")
    show(IOContext(io, :compact => true), support(gcal))
end


function prior_variable!(model, convexclass::AbstractSimplexPriorClass; var_name = "π") # adds constraints
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
    cdf_combination = _cdf.(likelihood_distribution.(Z, support(convexclass)), response(Z))
    @expression(model, dot(finite_param, cdf_combination))
end

function (target::LinearEBayesTarget)(prior::PriorVariable{<:DiscretePriorClass})
    @unpack convexclass, finite_param, model = prior
    linear_functional_evals = target.(support(convexclass))
    @expression(model, dot(finite_param, linear_functional_evals))
end



"""
    MixturePriorClass(components) <: Empirikos.ConvexPriorClass

Type representing the family of all mixture distributions with mixing components equal to
`components`, i.e., it represents all `MixtureModel` distributions with
`components = components` and `probs` taking values on the probability simplex.

Note that `MixturePriorClass(components)(probs) == MixtureModel(components, probs)`.

# Examples
```julia-repl
julia> gcal = MixturePriorClass([Normal(0,1), Normal(0,2)])
MixturePriorClass (K = 2)
Normal{Float64}(μ=0.0, σ=1.0)
Normal{Float64}(μ=0.0, σ=2.0)

julia> gcal([0.2,0.8])
MixtureModel{Normal{Float64}}(K = 2)
components[1] (prior = 0.2000): Normal{Float64}(μ=0.0, σ=1.0)
components[2] (prior = 0.8000): Normal{Float64}(μ=0.0, σ=2.0)
```
"""
struct MixturePriorClass{S} <: AbstractMixturePriorClass
    components::S
end

MixturePriorClass() = MixturePriorClass(nothing)
Distributions.components(mixclass::MixturePriorClass) = mixclass.components
function Distributions.component(mixclass::AbstractMixturePriorClass, i::Integer)
    Distributions.components(mixclass)[i]
end
nparams(mixclass::AbstractMixturePriorClass) = length(components(mixclass))
Distributions.ncomponents(mixclass::MixturePriorClass) = nparams(mixclass)
function (mixclass::AbstractMixturePriorClass)(p::AbstractVector{<:Real})
    MixtureModel(components(mixclass), fix_πs(p))
end

# Similar to show method for MixtureModel
function Base.show(io::IO, gcal::MixturePriorClass)
    K = ncomponents(gcal)
    println(io, "MixturePriorClass (K = $K)")
    Ks = min(K, 8)
    for i = 1:Ks
        println(io, Distributions.component(gcal, i))
    end
    if Ks < K
        println(io, "The rest are omitted ...")
    end
end

function pdf(prior::PriorVariable{<:AbstractMixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    pdf_combination = pdf.(components(convexclass), Z)
    @expression(model, dot(finite_param, pdf_combination))
end

function cdf(prior::PriorVariable{<:AbstractMixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    cdf_combination = _cdf.(components(convexclass), Z)
    @expression(model, dot(finite_param, cdf_combination))
end

function (target::LinearEBayesTarget)(prior::PriorVariable{<:AbstractMixturePriorClass})
    @unpack convexclass, finite_param, model = prior
    linear_functional_evals = target.(components(convexclass))
    @expression(model, dot(finite_param, linear_functional_evals))
end

"""
    GaussianScaleMixtureClass(σs) <: Empirikos.ConvexPriorClass

Type representing the family of mixtures of Gaussians with mean `0` and standard deviations
equal to `σs`. `GaussianScaleMixtureClass(σs)` represents the same class of distributions
as `MixturePriorClass.(Normal.(0, σs))`

```julia-repl
julia> gcal = GaussianScaleMixtureClass([1.0,2.0])
GaussianScaleMixtureClass | σs = [1.0, 2.0]

julia> gcal([0.2,0.8])
MixtureModel{Normal{Float64}}(K = 2)
components[1] (prior = 0.2000): Normal{Float64}(μ=0.0, σ=1.0)
components[2] (prior = 0.8000): Normal{Float64}(μ=0.0, σ=2.0)
```
"""
struct GaussianScaleMixtureClass{S} <: AbstractMixturePriorClass
    σs::S
end

GaussianScaleMixtureClass() = GaussianScaleMixtureClass(DataBasedDefault())

function Base.show(io::IO, gcal::GaussianScaleMixtureClass)
    print(io, "GaussianScaleMixtureClass | σs = ")
    show(IOContext(io, :compact => true), gcal.σs)
end

components(convexclass::GaussianScaleMixtureClass) = Normal.(0, convexclass.σs)
