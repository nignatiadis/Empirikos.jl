struct PoissonSample{T,S} <: DiscreteEBayesSample{T}
    Z::T
    E::S
end

PoissonSample(E) = BinomialSample(missing, E)
PoissonSample() = BinomialSample(missing, 1.0)

response(Z::PoissonSample) = Z.Z
nuisance_parameter(Z::PoissonSample) = Z.E

likelihood_distribution(Z::PoissonSample, λ) = Poisson(λ * nuisance_parameter(Z))

function Base.show(io::IO, Z::PoissonSample)
    spaces_to_keep = ismissing(response(Z)) ? 1 : max(3 - ndigits(response(Z)), 1)
    spaces = repeat(" ", spaces_to_keep)
    print(io, "Z=", response(Z), spaces, "| ", "E=", Z.E)
end

# how to break ties on n?
function Base.isless(a::PoissonSample, b::PoissonSample)
    a.E <= b.E && response(a) < response(b)
end

# Conjugate computations

function default_target_computation(::PoissonSample, ::Gamma)
    Conjugate()
end

function marginalize(Z::PoissonSample, prior::Gamma)
    E = nuisance_parameter(Z)
    @unpack α, θ = prior
    β = 1/θ
    p = β/(E+β)
    NegativeBinomial(α,p)
end

function posterior(Z::PoissonSample, prior::Gamma)
    E = nuisance_parameter(Z)
    @unpack α, θ = prior
    β = 1/θ
    α_post = α + response(Z)
    β_post = β + E
    Gamma(α_post, 1/β_post)
end

function StatsBase.fit(
    ::ParametricMLE{<:Gamma},
    Zs::AbstractVector{<:PoissonSample}
)
    func = TwiceDifferentiable(
        params -> -loglikelihood(Zs, Gamma(params...)),
        [1.0; 1.0];
        autodiff = :forward,
    )
    dfc = TwiceDifferentiableConstraints([0.0; 0.0], [Inf; Inf])

    opt = optimize(func, dfc, [1.0; 1.0], IPNewton())
    Gamma(Optim.minimizer(opt)...)
end


# DiscretePriorClass


function _set_defaults(convexclass::DiscretePriorClass,
    Zs::AbstractVector{<:PoissonSample};  #TODO for MultinomialSummary
    hints)
    eps = get(hints, :eps, 1e-4)
    prior_grid_length = get(hints, :prior_grid_length, 300)::Integer
    _sample_min, _sample_max =  extrema( response.(Zs) ./ nuisance_parameter.(Zs))
    _grid_min = max(2*eps, _sample_min - eps)
    _grid_max = _sample_max + eps
    DiscretePriorClass(range(_grid_min; stop=_grid_max, length=prior_grid_length))
end




#v <- seq(max(2 * eps, min(x/exposure)) - eps, max(x/exposure) + eps, length = v)
