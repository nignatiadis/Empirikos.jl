struct BinomialSample{T<:Integer,S<:Union{Missing,T}} <: DiscreteEBayesSample{T}
    Z::S
    n::T     # add checks that Z \in {0,...,n}
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
    skedasticity(Zs) == Homoskedastic() || error("Heteroskedastic likelihood not implemented.")
    _min, _max = extrema(response.(Zs))
    n = ntrials(Zs[1])
    BinomialSample.(_min:_max, n)
end

#-----------------------------------------------------------------------
#---------- Beta Binomial Conjugacy-------------------------------------
#-----------------------------------------------------------------------

function default_target_computation(::BinomialSample, ::Beta, ::AbstractPosteriorTarget)
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


function instantiate(convexclass::DiscretePriorClass{Nothing},
                     Zs::VectorOrSummary{<:BinomialSample};
                     kwargs...)
    eps = get(kwargs, :eps, 1e-4)
    prior_grid_length = get(kwargs, :prior_grid_length, 300)::Integer
    DiscretePriorClass(range(eps; stop=1-eps, length=prior_grid_length))
end
