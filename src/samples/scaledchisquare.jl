"""
    ScaledChiSquareSample(Z, ν)

An observed sample ``Z`` drawn from a scaled chi-square distribution with unknown scale ``\\sigma^2 > 0``.

```math
Z \\sim \\frac{\\sigma^2}{\\nu}}\\chi^2_{\\nu}
```

``\\sigma^2`` is assumed unknown. The type above is used when the sample ``Z`` is to be used for estimation or inference of ``\\sigma^2``.
"""
struct ScaledChiSquareSample{T, S} <: ContinuousEBayesSample{T}
    Z::T
    ν::S
end

ScaledChiSquareSample(ν) = ScaledChiSquareSample(missing, ν)


eltype(Z::ScaledChiSquareSample{T}) where {T} = T
support(Z::ScaledChiSquareSample) = RealInterval(0, +Inf)

response(Z::ScaledChiSquareSample) = Z.Z
nuisance_parameter(Z::ScaledChiSquareSample) = Z.ν
StatsBase.dof(Z::ScaledChiSquareSample) = Z.ν

function likelihood_distribution(Z::ScaledChiSquareSample, σ²)
    ν = Z.ν
    Gamma(ν/2,  σ² * 2 / ν)
end


function Base.show(io::IO, Z::ScaledChiSquareSample)
    resp_Z = response(Z)
    print(io, "ScaledΧ²(", resp_Z, "; σ², ν=", Z.ν,")")
end

struct InverseScaledChiSquare{T, S} <: ContinuousUnivariateDistribution
    σ²::T
    ν ::S
end

function Distributions.InverseGamma(d::InverseScaledChiSquare)
    _shape = d.ν/2
    _scale = d.ν/2 * d.σ²
    InverseGamma(_shape, _scale)
end

function InverseScaledChiSquare(d::Distributions.InverseGamma)
    _shape = shape(d)
    _scale = scale(d)
    ν = 2*_shape
    σ² = _scale / _shape
    InverseScaledChiSquare(σ², ν)
end

# Move all of the below to a macro?
function Distributions.pdf(d::InverseScaledChiSquare, x::Real)
   pdf(InverseGamma(d), x)
end


function Distributions.logpdf(d::InverseScaledChiSquare, x::T) where {T<:Real}
    logpdf(InverseGamma(d), x)
end


function Distributions.cdf(d::InverseScaledChiSquare, x::Real)
    cdf(InverseGamma(d), x)
end

function Distributions.logcdf(d::InverseScaledChiSquare, x::Real)
    Distributions.logcdf(InverseGamma(d), x)
end

function Distributions.mean(d::InverseScaledChiSquare)
    Distributions.mean(InverseGamma(d))
end

function Distributions.std(d::InverseScaledChiSquare)
    Distributions.std(InverseGamma(d))
end

function Distributions.quantile(d::InverseScaledChiSquare, p)
    Distributions.quantile(InverseGamma(d), p)
end

function Distributions.rand(rng::AbstractRNG, d::InverseScaledChiSquare)
    Distributions.rand(rng, InverseGamma(d))
end

# Conjugate computations

function default_target_computation(::BasicPosteriorTarget, ::ScaledChiSquareSample, ::InverseScaledChiSquare)
    Conjugate()
end

 function marginalize(Z::ScaledChiSquareSample, prior::InverseScaledChiSquare)
    σ² = prior.σ²
    Distributions.AffineDistribution{typeof(σ²)}(zero(σ²), σ², FDist(Z.ν, prior.ν))
end

function posterior(Z::ScaledChiSquareSample, prior::InverseScaledChiSquare)
    aggregate_ν = Z.ν + prior.ν
    aggregate_σ² = (Z.Z * Z.ν  +  prior.σ² * prior.ν) / aggregate_ν
    InverseScaledChiSquare(aggregate_σ², aggregate_ν)
end


function limma_pvalue(β_hat, ::ScaledChiSquareSample, prior::Dirac)
    σ = sqrt(prior.value)
    2*ccdf(Normal(0, σ), abs(β_hat))
end


function limma_pvalue(β_hat, Z::ScaledChiSquareSample, prior::InverseScaledChiSquare)
    post = posterior(Z, prior)
    t_moderated = β_hat / sqrt.(post.σ²)
    2*ccdf(TDist(post.ν), abs(t_moderated))
end


function limma_pvalue(β_hat, Z::ScaledChiSquareSample, prior::DiscreteNonParametric)
    post = posterior(Z, prior)
    σs = sqrt.(support(post))
    πs = probs(post)

    pvals = 2 .* ccdf.(Normal.(0, σs), (abs(β_hat)))
    LinearAlgebra.dot(pvals, πs)
end

function limma_pvalue(β_hat, Z::ScaledChiSquareSample, prior::MixtureModel)
    comps = components(prior)
    pvals_by_comp = limma_pvalue.(β_hat, Z, comps)
    post = posterior(Z, prior)
    πs_post = probs(post)
    LinearAlgebra.dot(pvals_by_comp, πs_post)
end



function fit_limma(Zs::AbstractVector{<:Empirikos.ScaledChiSquareSample})
    zs = response.(Zs)
    νs = dof.(Zs)
    logzs = log.(zs)

    es = logzs .- digamma.(νs/2) .+ log.(νs/2)
    μ_e = mean(es)
    var_e = Statistics.var(es; corrected=false)
    var_demeaned = var_e - mean(trigamma.(νs/2))

    if var_demeaned >0
        ν0 = 2invtrigamma(var_demeaned)
        σ² = exp(μ_e + digamma(ν0/2)-log(ν0/2))
    else
        ν0 = Inf
        σ² = mean(zs)
    end

    if ν0 == Inf
        prior = Dirac(σ²)
    else
        prior = InverseScaledChiSquare(σ², ν0)
    end
    prior
end


"""
    invtrigamma(x)
Compute the inverse [`trigamma`](@ref) function of `x`.
"""
invtrigamma(y::Number) = _invtrigamma(float(y))

function _invtrigamma(y::Float64)
    # Implementation of Newton algorithm described in
    # "Linear Models and Empirical Bayes Methods for Assessing
    #  Differential Expression in Microarray Experiments"
    # (Appendix "Inversion of Trigamma Function")
    #  by Gordon K. Smyth, 2004

    if y <= 0
        throw(DomainError(y, "Only positive `y` supported."))
    end

    if y > 1e7
        return inv(sqrt(y))
    elseif y < 1e-6
        return inv(y)
    end

    x_old = inv(y) + 0.5
    x_new = x_old

    # Newton iteration
    δ = Inf
    iteration = 0
    while δ > 1e-8 && iteration <= 25
        iteration += 1
        f_x_old = trigamma(x_old)
        δx =  f_x_old*(1-f_x_old/y) / polygamma(2, x_old)
        x_new = x_old + δx
        δ = - δx / x_new
        x_old = x_new
    end

    return x_new
end
