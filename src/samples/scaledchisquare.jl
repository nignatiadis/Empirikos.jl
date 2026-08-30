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
    halfνs = νs ./ 2

    es = logzs .- digamma.(halfνs) .+ log.(halfνs)
    μ_e = mean(es)
    var_e = Statistics.var(es; corrected = true)
    var_demeaned = var_e - mean(trigamma.(halfνs))

    if var_demeaned > 0
        ν0 = 2 * invtrigamma(var_demeaned)
        σ² = exp(μ_e + digamma(ν0 / 2) - log(ν0 / 2))
    else
        ν0 = Inf
        σ² = mean(zs)
    end

    isinf(ν0) ? Dirac(σ²) : InverseScaledChiSquare(σ², ν0)
end

function fit_limma_robust(
    Zs::AbstractVector{<:Empirikos.ScaledChiSquareSample};
    winsor_p = (0.05, 0.9),
)
    isempty(Zs) && throw(ArgumentError("Cannot fit limma prior to an empty vector."))
    νs = dof.(Zs)
    allequal(νs) || throw(ArgumentError("Robust limma requires equal degrees of freedom."))
    length(Zs) < 3 && return fit_limma(Zs)

    zs = response.(Zs)
    ν = first(νs)
    nonrobust_prior = fit_limma(Zs)
    ν0_nonrobust = nonrobust_prior isa Dirac ? Inf : nonrobust_prior.ν

    logzs = log.(zs)
    q = quantile(logzs, winsor_p)
    logzs_winsorized = clamp.(logzs, q[1], q[2])
    zwmean = mean(logzs_winsorized)
    zwvar = Statistics.var(logzs_winsorized; corrected = true)

    Z = ScaledChiSquareSample(ν)
    sampling_moments = _winsorized_log_moments(marginalize(Z, Dirac(1.0)), winsor_p)
    clipped_ν0 = if isinf(ν0_nonrobust) || zwvar <= sampling_moments.var
        Inf
    else
        _winsorized_dof(zwvar, ν, winsor_p, ν0_nonrobust)
    end
    # Winsorizing should only increase the prior information relative to the
    # ordinary limma moment fit; this matches limma's robust common-prior fit.
    ν0 = max(ν0_nonrobust, clipped_ν0)

    prior = isinf(ν0) ? Dirac(1.0) : InverseScaledChiSquare(1.0, ν0)
    mom = _winsorized_log_moments(marginalize(Z, prior), winsor_p)
    σ² = exp(zwmean - mom.mean)

    isinf(ν0) ? Dirac(σ²) : InverseScaledChiSquare(σ², ν0)
end

function fit_limma_quantile(
    Zs::AbstractVector{<:Empirikos.ScaledChiSquareSample};
    quantile_p = (0.2, 0.8),
    scale_p = 0.5,
)
    isempty(Zs) && throw(ArgumentError("Cannot fit limma prior to an empty vector."))
    νs = dof.(Zs)
    allequal(νs) || throw(ArgumentError("Quantile limma requires equal degrees of freedom."))
    length(Zs) < 3 && return fit_limma(Zs)

    zs = response.(Zs)
    ν = first(νs)

    q = quantile(zs, quantile_p)
    ν0 = _quantile_dof(q[2] / q[1], ν, quantile_p)
    prior = isinf(ν0) ? Dirac(1.0) : InverseScaledChiSquare(1.0, ν0)
    marginal = marginalize(ScaledChiSquareSample(ν), prior)
    σ² = quantile(zs, scale_p) / quantile(marginal, scale_p)

    isinf(ν0) ? Dirac(σ²) : InverseScaledChiSquare(σ², ν0)
end

function _winsorized_log_moments(marginal, winsor_p)
    pl, ph = winsor_p
    ql, qu = quantile.(Ref(marginal), winsor_p)
    zq = log.((ql, qu))
    pu = 1 - ph

    # Phipson-Smyth use fixed Gauss-Legendre quadrature here.
    # QuadGK evaluates the same integral using the package's existing
    # adaptive Gauss-Kronrod dependency.
    winsor_integral(f) = first(quadgk(ql, qu; rtol = 1e-8) do x
        f(log(x)) * pdf(marginal, x)
    end)

    integral1 = winsor_integral(identity)
    μ = integral1 + pl * zq[1] + pu * zq[2]

    integral2 = winsor_integral(z -> abs2(z - μ))
    v = integral2 + pl * abs2(zq[1] - μ) + pu * abs2(zq[2] - μ)

    (mean = μ, var = max(v, zero(v)))
end

function _winsorized_dof(zwvar, ν, winsor_p, ν0_min)
    Z = ScaledChiSquareSample(ν)
    f(logν0) = begin
        marginal = marginalize(Z, InverseScaledChiSquare(1.0, exp(logν0)))
        log(zwvar / _winsorized_log_moments(marginal, winsor_p).var)
    end
    lower = ν0_min
    flower = f(log(lower))
    isfinite(flower) || return ν0_min
    flower > 0 && return ν0_min
    upper = 2 * lower
    while isfinite(upper) && f(log(upper)) < 0
        upper *= 2
    end
    isfinite(upper) || return Inf
    exp(find_zero(f, log.((lower, upper)), Bisection()))
end

function _quantile_dof(ratio, ν, p)
    Z = ScaledChiSquareSample(ν)
    quantile_ratio(ν0) = begin
        prior = isinf(ν0) ? Dirac(1.0) : InverseScaledChiSquare(1.0, ν0)
        marginal = marginalize(Z, prior)
        quantile(marginal, p[2]) / quantile(marginal, p[1])
    end

    sampling_ratio = quantile_ratio(Inf)
    ratio <= sampling_ratio && return Inf

    f(logν0) = log(quantile_ratio(exp(logν0)) / ratio)

    lower = upper = one(float(ratio))
    fupper = f(log(upper))
    iszero(fupper) && return upper
    if fupper > 0
        while fupper > 0
            lower = upper
            upper *= 2
            isfinite(upper) || return Inf
            fupper = f(log(upper))
        end
    else
        flower = fupper
        while flower < 0
            upper = lower
            lower /= 2
            flower = f(log(lower))
            isfinite(flower) || throw(ArgumentError(
                "Quantile ratio implies prior degrees of freedom below the numerically supported range.",
            ))
        end
    end
    exp(find_zero(f, log.((lower, upper)), Bisection()))
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
