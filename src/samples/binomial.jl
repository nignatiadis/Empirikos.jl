"""
    BinomialSample(Z, n)

An observed sample ``Z`` drawn from a Binomial distribution with `n` trials.

```math
Z \\sim \\text{Binomial}(n, p)
```

``p`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``p``.

```jldoctest
julia> BinomialSample(2, 10)          # 2 out of 10 trials successful
Z=2  | n=10
```
"""
struct BinomialSample{T,S<:Integer} <: DiscreteEBayesSample{T}
    Z::T
    n::S     #TODO: add checks that Z \in {0,...,n}
end

BinomialSample(n::Integer) = BinomialSample(missing, n)

function Base.show(io::IO, Z::BinomialSample)
    spaces_to_keep = ismissing(response(Z)) ? 1 : max(3 - ndigits(response(Z)), 1)
    spaces = repeat(" ", spaces_to_keep)
    print(io, "Z=", response(Z), spaces, "| ", "n=", ntrials(Z))
end

function Base.show(io::IO, Z::BinomialSample{<:Interval})
    Zz = response(Z)
    print(io, "Z ∈ ")
    show(IOContext(io, :compact => true), Zz)
    print(io, " | ", "n=", ntrials(Z))
end

summarize_by_default(::AbstractVector{<:BinomialSample}) = true



response(Z::BinomialSample) = Z.Z
ntrials(Z::BinomialSample) = Z.n
nuisance_parameter(Z::BinomialSample) = ntrials(Z)

likelihood_distribution(Z::BinomialSample, p) = Binomial(ntrials(Z), p)


function fill_levels(Zs::AbstractVector{<:DiscreteEBayesSample})
    skedasticity(Zs) == Homoskedastic() ||
        error("Heteroskedastic likelihood not implemented.")
    #_min, _max = extrema(response.(Zs))
    #n = ntrials(Zs[1])
    #BinomialSample.(_min:_max, n)
    #BinomialSample.(0:n, n) #TODO!  Is it responsibility of whatever passes stuff in here
    # to e.g. include 0 counts?
    sort(unique(Zs))
end

function fill_levels(Zs::MultinomialSummary{<:DiscreteEBayesSample})
    fill_levels(collect(keys(Zs)))
end

function dictfun(::Nothing, Zs_summary::VectorOrSummary{<:DiscreteEBayesSample}, f)
    skedasticity(Zs_summary) == Homoskedastic() ||
        error("Heteroskedastic likelihood not implemented.")
    Zs = fill_levels(Zs_summary)
    if !isa(f, AbstractVector)
        f = f.(Zs)
    end
    DictFunction(Zs, f)
end


#-----------------------------------------------------------------------
#---------- Beta Binomial Conjugacy-------------------------------------
#-----------------------------------------------------------------------

function default_target_computation(::BasicPosteriorTarget,::BinomialSample, ::Beta)
    Conjugate()
end

function marginalize(Z::BinomialSample, prior::Beta)
    @unpack α, β = prior
    BetaBinomial(ntrials(Z), α, β)
end

function posterior(Z::BinomialSample, prior::Beta)
    Beta(prior.α + response(Z), prior.β + ntrials(Z) - response(Z))
end

# Fit BetaBinomial
function StatsBase.fit(
    ::MethodOfMoments{<:Beta},
    Zs::VectorOrSummary{<:BinomialSample},
    ::Homoskedastic,
)
    # TODO: Let ::Homoskedastic carry type information.
    n = ntrials(Zs[1])
    μ₁ = mean(response.(Zs), weights(Zs))
    μ₂ = mean(abs2.(response.(Zs)), weights(Zs))
    denom = n * (μ₂ / μ₁ - μ₁ - 1) + μ₁
    α = (n * μ₁ - μ₂) / denom
    β = (n - μ₁) * (n - μ₂ / μ₁) / denom
    Beta(α, β)
end





# DiscretePriorClass


function _set_defaults(
    convexclass::DiscretePriorClass,
    Zs::VectorOrSummary{<:BinomialSample};
    hints,
)
    eps = get(hints, :eps, 1e-4)
    prior_grid_length = get(hints, :prior_grid_length, 300)::Integer
    DiscretePriorClass(range(eps; stop = 1 - eps, length = prior_grid_length))
end


"""
    ChiSquaredFLocalization(α) <: FLocalization

The ``\\chi^2`` F-localization at confidence level ``1-\\alpha`` for a discrete random variable
taking values in ``\\{0,\\dotsc, N\\}``. It is equal to:
```math
f: \\sum_{x=0}^N \\frac{(n \\hat{f}_n(x) - n f(x))^2}{n f(x)} \\leq \\chi^2_{N,1-\\alpha},
```
where ``\\chi^2_{N,1-\\alpha}`` is the ``1-\\alpha`` quantile of the Chi-squared
distribution with ``N`` degrees of freedom, ``n`` is the sample size,
``\\hat{f}_n(x)`` is the proportion of samples equal to ``x`` and ``f(x)`` is then
population pmf.
"""
Base.@kwdef struct ChiSquaredFLocalization{T} <: FLocalization
    α::T = 0.05
end

vexity(::ChiSquaredFLocalization) = ConvexVexity()


struct FittedChiSquaredFLocalization{T,S,D<:AbstractDict{T,S},C} <: FittedFLocalization
    summary::D
    band::S
    chisq::C
    dof::Int
    n::Int
end

vexity(chisq::FittedChiSquaredFLocalization) = vexity(chisq.chisq)

function nominal_alpha(chisq::FittedChiSquaredFLocalization)
    nominal_alpha(chisq.chisq)
end

# TODO: Allow this to work more broadly.
function StatsBase.fit(chisq::ChiSquaredFLocalization, Zs::AbstractVector{<:BinomialSample})
    StatsBase.fit(chisq, summarize(Zs))
end

function StatsBase.fit(chisq::ChiSquaredFLocalization, Zs_summary::MultinomialSummary)
    n = nobs(Zs_summary)
    _levels = fill_levels(Zs_summary)
    _dof = length(_levels) - 1        #ntrials(_levels[1]) #again maybe Homoskedastic() should return what type of homoskedastic
    empirical_probs = Zs_summary.(_levels) ./ n
    _dict = SortedDict(keys(Zs_summary.store) .=> empirical_probs)
    α = nominal_alpha(chisq)
    band =  quantile(Chisq(_dof), 1-α)
    FittedChiSquaredFLocalization(_dict, band, chisq, _dof, n)
end



function flocalization_constraint!(
    model,
    chisq::FittedChiSquaredFLocalization,
    prior::PriorVariable,
)
    n = chisq.n

    ts = @variable(model, [1:(chisq.dof+1)])
    @constraint(model, ts .>= 0)
    band = chisq.band
    for (i, (Z, pdf_value)) in enumerate(chisq.summary)
        _pdf = pdf(prior, Z::EBayesSample)
        @constraint(model, [ts[i]; n * _pdf; n * _pdf - n * pdf_value] in RotatedSecondOrderCone())
    end
    @constraint(model, sum(ts) <= band/2)
    model
end
