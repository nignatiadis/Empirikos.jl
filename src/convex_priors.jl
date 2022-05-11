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

abstract type AbstractMixturePriorClass <: ConvexPriorClass end

# TODO: implement correct projection onto simplex and check deviation is not too big
# though this should mostly help with minor numerical difficulties
function fix_πs(πs)
    πs = max.(πs, 0.0)
    πs = πs ./ sum(πs)
end

function Distributions.component(mixclass::AbstractMixturePriorClass, i::Integer)
    Distributions.components(mixclass)[i]
end
Distributions.ncomponents(mixclass::AbstractMixturePriorClass) = length(components(mixclass))
nparams(mixclass::AbstractMixturePriorClass) = Distributions.ncomponents(mixclass)

function (mixclass::AbstractMixturePriorClass)(p::AbstractVector{<:Real})
    MixtureModel(components(mixclass), fix_πs(p))
end

function pdf(prior::PriorVariable{<:AbstractMixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    pdf_combination = pdf.(components(convexclass), Z)
    @expression(model, dot(finite_param, pdf_combination))
end

function cdf(prior::PriorVariable{<:AbstractMixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    cdf_combination = cdf.(components(convexclass), Z)
    @expression(model, dot(finite_param, cdf_combination))
end

function (target::LinearEBayesTarget)(prior::PriorVariable{<:AbstractMixturePriorClass})
    @unpack convexclass, finite_param, model = prior
    linear_functional_evals = target.(components(convexclass))
    @expression(model, dot(finite_param, linear_functional_evals))
end

function prior_variable!(model, convexclass::AbstractMixturePriorClass)
    n = nparams(convexclass)
    tmp_vars = @variable(model, [i = 1:n])
    set_lower_bound.(tmp_vars, 0.0)
    con = @constraint(model, sum(tmp_vars) == 1.0)
    PriorVariable(convexclass, tmp_vars, model)
end




"""
    DiscretePriorClass(support) <: Empirikos.ConvexPriorClass

Type representing the family of all discrete distributions supported on a subset
of `support`, i.e., it represents all `DiscreteNonParametric` distributions with
`support = support` and `probs` taking values on the probability simplex.

Note that `DiscretePriorClass(support)(probs) == DiscreteNonParametric(support, probs)`.

# Examples
```jldoctest
julia> gcal = DiscretePriorClass([0,0.5,1.0])
DiscretePriorClass | support = [0.0, 0.5, 1.0]

julia> gcal([0.2,0.2,0.6])
DiscreteNonParametric{Float64, Float64, Vector{Float64}, Vector{Float64}}(support=[0.0, 0.5, 1.0], p=[0.2, 0.2, 0.6])
```
"""
struct DiscretePriorClass{S} <: AbstractMixturePriorClass
    support::S
end

DiscretePriorClass(; support=DataBasedDefault()) = DiscretePriorClass(support)

support(convexclass::DiscretePriorClass) = convexclass.support

function (convexclass::DiscretePriorClass)(p::AbstractVector{<:Real})
    DiscreteNonParametric(support(convexclass), fix_πs(p))
end

Distributions.components(convexclass::DiscretePriorClass) = Dirac.(support(convexclass))

function Base.show(io::IO, gcal::DiscretePriorClass)
    print(io, "DiscretePriorClass | support = ")
    show(IOContext(io, :compact => true), support(gcal))
end



"""
    SymmetricDiscretePriorClass(support) <: Empirikos.ConvexPriorClass

Type representing the family of all symmetric discrete distributions supported on a subset
of `support`∩`-support`, i.e., it represents all `DiscreteNonParametric` distributions with
`support = [support;-support]` and `probs` taking values on the probability simplex
(so that components with same magnitude, but opposite sign have the same probability).
`support` should include the nonnegative support points only.
"""
struct SymmetricDiscretePriorClass{S} <: AbstractMixturePriorClass
    support::S
end

SymmetricDiscretePriorClass(; support=DataBasedDefault()) = SymmetricDiscretePriorClass(support)

support(convexclass::SymmetricDiscretePriorClass) = [-convexclass.support; convexclass.support]

function (convexclass::SymmetricDiscretePriorClass)(p::AbstractVector{<:Real})
    DiscreteNonParametric(support(convexclass), [fix_πs(p)/2; fix_πs(p)/2] )
end

# The below is different than the default option. TODO: Test
nparams(convexclass::SymmetricDiscretePriorClass) = length(convexclass.support)
function Distributions.components(convexclass::SymmetricDiscretePriorClass)
    [MixtureModel([Dirac(-x);Dirac(x)], [1/2; 1/2]) for x in convexclass.support]
end

function Base.show(io::IO, gcal::SymmetricDiscretePriorClass)
    print(io, "SymmetricDiscretePriorClass | nonneg. support = ")
    show(IOContext(io, :compact => true), gcal.support)
end





"""
    MixturePriorClass(components) <: Empirikos.ConvexPriorClass

Type representing the family of all mixture distributions with mixing components equal to
`components`, i.e., it represents all `MixtureModel` distributions with
`components = components` and `probs` taking values on the probability simplex.

Note that `MixturePriorClass(components)(probs) == MixtureModel(components, probs)`.

# Examples

```jldoctest
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



"""
    GaussianScaleMixtureClass(σs) <: Empirikos.ConvexPriorClass

Type representing the family of mixtures of Gaussians with mean `0` and standard deviations
equal to `σs`. `GaussianScaleMixtureClass(σs)` represents the same class of distributions
as `MixturePriorClass.(Normal.(0, σs))`

```jldoctest
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
