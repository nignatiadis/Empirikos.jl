struct BinomialSample{T,S<:Integer} <: DiscreteEBayesSample{T}
    Z::T
    n::S     # add checks that Z \in {0,...,n}
end

BinomialSample(n::Integer) = BinomialSample(missing, n)

function Base.show(io::IO, Z::BinomialSample)
    spaces_to_keep = ismissing(response(Z)) ? 1 : max(3 - ndigits(response(Z)), 1)
    spaces = repeat(" ", spaces_to_keep)
    print(io, "Z=", response(Z), spaces, "| ", "n=", ntrials(Z))
end

summarize_by_default(::AbstractVector{<:BinomialSample}) = true

# how to break ties on n?
function Base.isless(a::BinomialSample, b::BinomialSample)
    ntrials(a) <= ntrials(b) && response(a) < response(b)
end

function Base.isless(a, b::BinomialSample)
    a < response(b)
end
function <=(a, b::BinomialSample)
    a <= response(b)
end
function Base.isless(a::BinomialSample, b)
    response(a) < b
end
function <=(a::BinomialSample, b)
    response(a) <= b
end


response(Z::BinomialSample) = Z.Z
ntrials(Z::BinomialSample) = Z.n
nuisance_parameter(Z::BinomialSample) = ntrials(Z)

likelihood_distribution(Z::BinomialSample, p) = Binomial(ntrials(Z), p)


function fill_levels(Zs::AbstractVector{<:BinomialSample})
    skedasticity(Zs) == Homoskedastic() ||
        error("Heteroskedastic likelihood not implemented.")
    #_min, _max = extrema(response.(Zs))
    n = ntrials(Zs[1])
    #BinomialSample.(_min:_max, n)
    BinomialSample.(0:n, n)
end

function fill_levels(Zs::MultinomialSummary{<:BinomialSample})
    fill_levels(collect(keys(Zs)))
end

function dictfun(::Nothing, Zs_summary::MultinomialSummary{<:BinomialSample}, f)
    skedasticity(Zs_summary) == Homoskedastic() ||
        error("Heteroskedastic likelihood not implemented.")
    Zs = fill_levels(collect(keys(Zs_summary)))
    if !isa(f, AbstractVector)
        f = f.(Zs)
    end
    DictFunction(Zs, f)
end

# TODO: Fix repetition with code above. Maybe separate fill_levels?
function dictfun(::Nothing, Zs_summary::AbstractVector{<:BinomialSample}, f)
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

function default_target_computation(::BinomialSample, ::Beta)
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

function StatsBase.fit(
    method::ParametricMLE{<:Beta},
    Zs::AbstractVector{<:BinomialSample},
    skedasticity,
)
    StatsBase.fit(method, summarize(Zs), skedasticity)
end

function StatsBase.fit(
    ::ParametricMLE{<:Beta},
    Zs_summary::MultinomialSummary{<:BinomialSample},
    skedasticity,
)
    func = TwiceDifferentiable(
        params -> -loglikelihood(Zs_summary, Beta(params...)),
        [1.0; 1.0];
        autodiff = :forward,
    )
    dfc = TwiceDifferentiableConstraints([0.0; 0.0], [Inf; Inf])

    opt = optimize(func, dfc, [1.0; 1.0], IPNewton())
    Beta(Optim.minimizer(opt)...)
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


# Chi-squared neighborhood



Base.@kwdef struct ChiSquaredNeighborhood{T} <: EBayesNeighborhood
    α::T = 0.05
end

vexity(::ChiSquaredNeighborhood) = ConvexVexity()


struct FittedChiSquaredNeighborhood{T,S,D<:AbstractDict{T,S},C} <: FittedEBayesNeighborhood
    summary::D
    band::S
    chisq::C
    dof::Int
    n::Int
end

vexity(chisq::FittedChiSquaredNeighborhood) = vexity(chisq.chisq)

function nominal_alpha(chisq::FittedChiSquaredNeighborhood)
    nominal_alpha(chisq.chisq)
end

# TODO: Allow this to work more broadly.
function StatsBase.fit(chisq::ChiSquaredNeighborhood, Zs::AbstractVector{<:BinomialSample})
    StatsBase.fit(chisq, summarize(Zs))
end

function StatsBase.fit(chisq::ChiSquaredNeighborhood, Zs_summary::MultinomialSummary)
    n = nobs(Zs_summary)
    _levels = fill_levels(Zs_summary)
    _dof = ntrials(_levels[1]) #again maybe Homoskedastic() should return what type of homoskedastic
    empirical_probs = Zs_summary.(_levels) ./ n
    _dict = SortedDict(keys(Zs_summary.store) .=> empirical_probs)
    α = nominal_alpha(chisq)
    band =  quantile(Chisq(_dof), 1-α)
    FittedChiSquaredNeighborhood(_dict, band, chisq, _dof, n)
end



function neighborhood_constraint!(
    model,
    chisq::FittedChiSquaredNeighborhood,
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
