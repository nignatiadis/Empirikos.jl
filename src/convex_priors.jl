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

function rescaled_pdf(prior::PriorVariable{<:AbstractMixturePriorClass}, Z::EBayesSample)
    @unpack convexclass, finite_param, model = prior
    logpdf_combination = logpdf.(components(convexclass), Z)
    pdf_combination = exp.(logpdf_combination .- maximum(logpdf_combination))
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

DiscretePriorClass(; support=nothing) = DiscretePriorClass(support)

support(convexclass::DiscretePriorClass) = convexclass.support

function (convexclass::DiscretePriorClass)(p::AbstractVector{<:Real})
    DiscreteNonParametric(support(convexclass), fix_πs(p))
end

Distributions.components(convexclass::DiscretePriorClass) = Dirac.(support(convexclass))

function Base.show(io::IO, gcal::DiscretePriorClass)
    print(io, "DiscretePriorClass | support = ")
    show(IOContext(io, :compact => true), support(gcal))
end

function clean(G::DiscreteNonParametric; tol = 10*sqrt(eps()))
    πs = probs(G)
    non_zero_idx = πs .> tol
    new_πs = fix_πs( πs[non_zero_idx])
    DiscreteNonParametric(support(G)[non_zero_idx], new_πs)
end

"""
    SymmetricDiscretePriorClass(support) <: Empirikos.ConvexPriorClass

Type representing the family of all symmetric discrete distributions supported on a subset
of `support`∩`-support`, i.e., it represents all `DiscreteNonParametric` distributions with
`support = [support;-support]` and `probs` taking values on the probability simplex
(so that components with same magnitude, but opposite sign have the same probability).
`support` should include the nonnegative support points only.
"""
Base.@kwdef struct SymmetricDiscretePriorClass{S} <: AbstractMixturePriorClass
    support::S = nothing
    includes_zero::Bool = iszero(support[1])
end

function support(convexclass::SymmetricDiscretePriorClass)
    nonneg_supp = convexclass.support
    if convexclass.includes_zero
        supp = [-nonneg_supp[2:end]; nonneg_supp[1]; nonneg_supp[2:end]]
    else
        supp = [-nonneg_supp; nonneg_supp]
    end
    supp
end

function (convexclass::SymmetricDiscretePriorClass)(p::AbstractVector{<:Real})
    fixed_p = fix_πs(p)
    if convexclass.includes_zero
        πs = [fixed_p[2:end]/2; fixed_p[1]; fixed_p[2:end]/2]
    else
        πs = [fixed_p/2; fixed_p/2]
    end
    d = DiscreteNonParametric(support(convexclass), πs)
end

# The below is different than the default option. TODO: Test
nparams(convexclass::SymmetricDiscretePriorClass) = length(convexclass.support)

function Distributions.components(convexclass::SymmetricDiscretePriorClass)
    if convexclass.includes_zero
        comps =  [MixtureModel([Dirac(convexclass.support[1])], [1.0]);
            [MixtureModel([Dirac(-x);Dirac(x)], [1/2; 1/2]) for x in convexclass.support[2:end]]]
    else
        comps = [MixtureModel([Dirac(-x);Dirac(x)], [0.5; 0.5]) for x in convexclass.support]
    end
    comps
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

GaussianScaleMixtureClass() = GaussianScaleMixtureClass(nothing)

function Base.show(io::IO, gcal::GaussianScaleMixtureClass)
    print(io, "GaussianScaleMixtureClass | σs = ")
    show(IOContext(io, :compact => true), gcal.σs)
end

components(convexclass::GaussianScaleMixtureClass) = Normal.(0, convexclass.σs)


struct BetaMixtureClass{S} <: AbstractMixturePriorClass
    αs::S
    βs::S
end

BetaMixtureClass() = BetaMixtureClass(nothing, nothing)

components(convexclass::BetaMixtureClass) = Beta.(convexclass.αs, convexclass.βs)

"""
    UniformScaleMixtureClass(as) <: Empirikos.ConvexPriorClass

Type representing the family of symmetric uniform mixtures with bounds `[-a, a]`.
`UniformScaleMixtureClass(as)` represents the same class of distributions as
`MixturePriorClass.(Uniform.(-a, a))` for a in `as`.
```jldoctest
julia> ucal = UniformScaleMixtureClass([1.0,2.0])
UniformScaleMixtureClass | as = [1.0, 2.0]

julia> ucal([0.2,0.8])
MixtureModel{Uniform{Float64}}(K = 2)
components[1] (prior = 0.2000): Uniform{Float64}(a=-1.0, b=1.0)
components[2] (prior = 0.8000): Uniform{Float64}(a=-2.0, b=2.0)
```    
"""
struct UniformScaleMixtureClass{S} <: AbstractMixturePriorClass
    as::S 
    end

UniformScaleMixtureClass() = UniformScaleMixtureClass(nothing)

function Base.show(io::IO, ucal::UniformScaleMixtureClass)
    print(io, "UniformScaleMixtureClass | as = ")
    show(IOContext(io, :compact => true), ucal.as)
end
    
components(convexclass::UniformScaleMixtureClass) = [Uniform(-a, a) for a in convexclass.as]
    
"""
    GaussianLocationScaleMixtureClass(μs,std, σs) <: Empirikos.ConvexPriorClass

Type representing a mixture of two types of Gaussians:
1. Components with means `μs` and fixed standard deviation `std`
2. Components with mean `0` and standard deviations `σs`
`GaussianLocationScaleMixtureClass(μs,std, σs)` represents the same class of distributions
as `MixturePriorClass.(Normal.(0, σs))`
```jldoctest
julia> Glscal = GaussianLocationScaleMixtureClass([1.0,2.0], 0.05, [1,2])
GaussianLocationScaleMixtureClass | μs = [1.0, 2.0], std = 0.05, σs = [1, 2]

julia> Glscal([0.2,0.3,0.4,0.1])
MixtureModel{Normal{Float64}}(K = 4)
components[1] (prior = 0.2000): Normal{Float64}(μ=1.0, σ=0.05)
components[2] (prior = 0.3000): Normal{Float64}(μ=2.0, σ=0.05)
components[3] (prior = 0.4000): Normal{Float64}(μ=0.0, σ=1.0)
components[4] (prior = 0.1000): Normal{Float64}(μ=0.0, σ=2.0)
```    
"""
struct GaussianLocationScaleMixtureClass{M,T,S} <: AbstractMixturePriorClass
    μs::M
    std::T
    σs::S
end

GaussianLocationScaleMixtureClass() = GaussianLocationScaleMixtureClass(nothing, nothing, nothing)
function Base.show(io::IO, gcal::GaussianLocationScaleMixtureClass)
    print(io, "GaussianLocationScaleMixtureClass | μs = ")
    show(IOContext(io, :compact => true), gcal.μs)
    print(io, ", std = ", gcal.std, ", σs = ")
    show(IOContext(io, :compact => true), gcal.σs)
end

function components(gcal::GaussianLocationScaleMixtureClass)
  location_comps = Normal.(gcal.μs, gcal.std)
  zero_comps = Normal.(0, gcal.σs)
  vcat(location_comps, zero_comps)
end   

