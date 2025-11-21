abstract type InfiniteOrderKernel <: ContinuousUnivariateDistribution end

function Base.show(io::IO, d::InfiniteOrderKernel)
    print(io, nameof(typeof(d)))
    print(io, " | bandwidth = ")
    print(io, d.h)
end

"""
    SincKernel(h) <: InfiniteOrderKernel

Implements the `SincKernel` with bandwidth `h` to be used for kernel density estimation
through the `KernelDensity.jl` package. The sinc kernel is defined as follows:
```math
K_{\\text{sinc}}(x) = \\frac{\\sin(x)}{\\pi x}
```
It is not typically used for kernel density estimation, because this kernel is not
a density itself. However, it is particularly well suited to deconvolution problems
and estimation of very smooth densities because its Fourier transform is the following:
```math
K^*_{\\text{sinc}}(t) = \\mathbf 1( t \\in [-1,1])
```
"""
struct SincKernel{H} <: InfiniteOrderKernel
    h::H
end

SincKernel() = SincKernel(nothing)

bandwidth(kernel::InfiniteOrderKernel) = kernel.h 

function default_bandwidth(::InfiniteOrderKernel, 
    Zs::Union{AbstractVector{<:Empirikos.AbstractNormalSample}, AbstractVector{<:Empirikos.FoldedNormalSample}})
    skedasticity(Zs) == Homoskedastic() ||
        throw("Only implemented for Homoskedastic Gaussian data.")
    σ = std(Zs[1])
    n = length(Zs)
    h = σ / sqrt(log(n))
    h
end 

Distributions.cf(a::SincKernel, t) = one(Float64) * (-1 / a.h <= t <= 1 / a.h)

function Distributions.pdf(a::SincKernel, t::Real)
    if t == zero(Float64)
        return (one(Float64) / pi / a.h)
    else
        return (sin(t / a.h) / pi / t)
    end
end


"""
    DeLaValleePoussinKernel(h) <: InfiniteOrderKernel

Implements the `DeLaValleePoussinKernel` with bandwidth `h` to be used for kernel density estimation
through the `KernelDensity.jl` package. The De La Vallée-Poussin kernel is defined as follows:
```math
K_V(x) = \\frac{\\cos(x)-\\cos(2x)}{\\pi x^2}
```
Its use case is similar to the [`SincKernel`](@ref), however it has the advantage of being integrable
(in the Lebesgue sense) and having bounded total variation. Its Fourier transform is the following:
```math
K^*_V(t) = \\begin{cases}
 1, & \\text{ if } |t|\\leq 1 \\\\
 0, &\\text{ if } |t| \\geq 2 \\\\
 2-|t|,& \\text{ if } |t| \\in [1,2]
 \\end{cases}
```
"""
struct DeLaValleePoussinKernel{H} <: InfiniteOrderKernel
    h::H
end

DeLaValleePoussinKernel() = DeLaValleePoussinKernel(nothing)

function Distributions.cf(a::DeLaValleePoussinKernel, t)
    if abs(t * a.h) <= 1
        return (one(Float64))
    elseif abs(t * a.h) <= 2
        return (2 * one(Float64) - abs(t * a.h))
    else
        return (zero(Float64))
    end
end

function Distributions.pdf(a::DeLaValleePoussinKernel, t::Real)
    if t == zero(Float64)
        return (3 * one(Float64) / 2 / pi / a.h)
    else
        return (a.h * (cos(t / a.h) - cos(2 * t / a.h)) / pi / t^2)
    end
end



"""
    FlatTopKernel(h) < InfiniteOrderKernel

Implements the `FlatTopKernel` with bandwidth `h` to be used for kernel density estimation
through the `KernelDensity.jl` package. The flat-top kernel is defined as follows:
```math
K(x) = \\frac{\\sin^2(1.1x/2)-\\sin^2(x/2)}{\\pi x^2/ 20}.
```
Its use case is similar to the [`SincKernel`](@ref), however it has the advantage of being integrable
(in the Lebesgue sense) and having bounded total variation. Its Fourier transform is the following:
```math
K^*(t) = \\begin{cases}
 1, & \\text{ if } t|\\leq 1 \\\\
 0, &\\text{ if } |t| \\geq 1.1 \\\\
 11-10|t|,& \\text{ if } |t| \\in [1,1.1]
 \\end{cases}
```

```jldoctest
julia> Empirikos.FlatTopKernel(0.1)
FlatTopKernel | bandwidth = 0.1
```
"""
struct FlatTopKernel{H} <: InfiniteOrderKernel
    h::H
end

FlatTopKernel() = FlatTopKernel(nothing)

function Distributions.cf(a::FlatTopKernel, t)
    if abs(t * a.h) <= 1
        return (one(Float64))
    elseif abs(t * a.h) <= 1.1
        return (one(Float64) - 10 * (abs(t * a.h) - one(Float64)))
    else
        return (zero(Float64))
    end
end

# TODO: This is *not* numerically stable.
# But presumably do not need it to be (since the KDE code just calls cf)
function Distributions.pdf(a::FlatTopKernel, t::Real)
    h = a.h
    th = t / h
    if t == zero(Float64)
        return ((0.55^2 - 0.5^2) * 20 / pi / h)
    else
        return ((abs2(sin(0.55 * th)) - abs2(sin(0.5 * th))) / pi / th^2 * 20 / h)
    end
end


"""
    InfinityNormDensityBand(;a_min,
                             a_max,
                             kernel  =  Empirikos.FlatTopKernel(),
                             bootstrap = :Multinomial,
                             nboot = 1000,
                             α = 0.05,
                             rng = Random.MersenneTwister(1)
                        )  <: FLocalization


This struct contains hyperparameters that will be used for constructing a neighborhood
of the marginal density. The steps of the method (and corresponding hyperparameter meanings)
are as follows
* First a kernel density estimate ``\\bar{f}`` with `kernel` is fit to the data.
* Second, a `bootstrap` (options: `:Multinomial` or `Poisson`) with `nboot` bootstrap replicates will be used to estimate ``c_n``, such that:
```math
\\liminf_{n \\to \\infty}\\mathbb{P}\\left[\\sup_{x \\in [a_{\\text{min}} , a_{\\text{max}}]} | \\bar{f}(x) - f(x)| \\leq c_ n\\right] \\geq 1-\\alpha
```
Note that the bound is valid from `a_min` to `a_max`. `α` is the nominal level and finally
`rng` sets the seed for the bootstrap samples.
"""
Base.@kwdef struct InfinityNormDensityBand <: FLocalization
    a_min = nothing
    a_max = nothing
    npoints::Integer = 1024
    kernel = FlatTopKernel()
    bootstrap = :Multinomial
    nboot::Integer = 1000
    α::Float64 = 0.05
    rng = Random.MersenneTwister(1)
end

function Base.show(io::IO, d::InfinityNormDensityBand)
    print(io, "∞-density band [α: ",d.α, "] [Kernel: ")
    Base.show(io, d.kernel)
    print(io, "] [Bootstrap: ", string(d.bootstrap),"(",d.nboot,")]")
end

Empirikos.vexity(::InfinityNormDensityBand) = Empirikos.LinearVexity()


function flocalization_default_data_range(Zs::AbstractVector{<:Empirikos.AbstractNormalSample{<:Number}})
    skedasticity(Zs) == Homoskedastic() ||
        throw("Only implemented for Homoskedastic Gaussian data.")
    q = 0.005
    quantile(response.(Zs), (q, 1 - q))
end

function flocalization_default_data_range(Zs::AbstractVector{<:Empirikos.FoldedNormalSample{<:Number}})
    skedasticity(Zs) == Homoskedastic() ||
        throw("Only implemented for Homoskedastic Gaussian data.")
    q = 0.005
    a_max = quantile(abs.(response.(Zs)), 1-q/2)
    a_min = -a_max
    a_min, a_max
end

"""
    FittedInfinityNormDensityBand

The result of running
```julia
StatsBase.fit(opt::InfinityNormDensityBand, Zs)
```
Here `opt` is an instance
of [`InfinityNormDensityBand`](@ref) and `Zs` is a vector of [`AbstractNormalSample`](@ref)s
distributed according to a density ``f``..

## Fields:
* `a_min`,`a_max`, `kernel`: These are the same as the fields in `opt::InfinityNormDensityBand`.
* `C∞`: The half-width of the L∞ band.
* `fitted_kde`: The fitted `KernelDensity` object.
"""
Base.@kwdef struct FittedInfinityNormDensityBand{T<:Real,S,K} <: FittedFLocalization
    C∞::T
    a_min::T
    a_max::T
    fitted_kde::Any
    interp_kde::Any
    midpoints::S
    estimated_density::K
    boot_samples::Any
    method = nothing
end

Empirikos.vexity(::FittedInfinityNormDensityBand) = Empirikos.LinearVexity()

function nominal_alpha(inftyband::FittedInfinityNormDensityBand)
    nominal_alpha(inftyband.method)
end

function StatsBase.fit(
    opt::InfinityNormDensityBand,
    Zs::AbstractVector{<:Empirikos.AbstractNormalSample{<:Number}};
    kwargs...,
)
    skedasticity(Zs) == Homoskedastic() ||
        throw("Only implemented for Homoskedastic Gaussian data.")

    if isnothing(opt.a_min) && isnothing(opt.a_max)
        a_min, a_max = flocalization_default_data_range(Zs)
        opt = @set opt.a_min = a_min
        opt = @set opt.a_max = a_max
    end 

    if isnothing(bandwidth(opt.kernel))
        opt = @set opt.kernel.h = default_bandwidth(opt.kernel, Zs)
    end

    (; a_min, a_max, npoints, kernel, nboot, α, bootstrap, rng) = opt

    # deepcopying below to make sure RNG status for sampling here remains the same.
    rng = deepcopy(rng)

    res = certainty_banded_KDE(
        response.(Zs),
        a_min,
        a_max;
        npoints = npoints,
        rng = rng,
        kernel = kernel,
        bootstrap = bootstrap,
        nboot = nboot,
        α = α,
    )
    res = @set res.method = opt
    Z = Zs[1]
    normal_midpoints = [@set Z.Z = mdpt for mdpt in res.midpoints]
    res = @set res.midpoints = normal_midpoints
    res
end


# TODO: handle this at same time as the StandardNormalSample case.
function StatsBase.fit(
    opt::InfinityNormDensityBand,
    Zs::AbstractVector{<:Empirikos.FoldedNormalSample{<:Number}};
    kwargs...,
)
    skedasticity(Zs) == Homoskedastic() ||
        throw("Only implemented for Homoskedastic Gaussian data.")

    if isnothing(opt.a_min) && isnothing(opt.a_max)
        a_min, a_max = flocalization_default_data_range(Zs)
        opt = @set opt.a_min = a_min
        opt = @set opt.a_max = a_max
    end 

    if isnothing(bandwidth(opt.kernel))
        opt = @set opt.kernel.h = default_bandwidth(opt.kernel, Zs)
    end
    (; a_min, a_max, npoints, kernel, nboot, α, bootstrap, rng) = opt

    # deepcopying below to make sure RNG status for sampling here remains the same.
    rng = deepcopy(rng)

    random_signs = 2 .* rand(rng, Bernoulli(), length(Zs)) .-1

    res = certainty_banded_KDE(
        random_signs .* response.(Zs),
        a_min,
        a_max;
        absolute_value = true,
        npoints = 2*npoints,
        rng = rng,
        kernel = kernel,
        bootstrap = bootstrap,
        nboot = nboot,
        α = α,
    )
    res = @set res.method = opt
    Z = Zs[1]
    normal_midpoints = [@set Z.Z = mdpt for mdpt in res.midpoints]
    res = @set res.midpoints = normal_midpoints
    res = @set res.a_min = 0.0
    res
end


function certainty_banded_KDE(
    Xs,
    a_min,
    a_max;
    kernel,
    absolute_value = false,
    rng = Random._GLOBAL_RNG,
    npoints = 4096,
    nboot = 1_000,
    bootstrap = :Multinomial,
    α = 0.5,
)

    absolute_value && (a_min != -a_max) && error("a_min needs to be equal to a_max")

    h = kernel.h
    m = length(Xs)

    if absolute_value
        hi = maximum(abs.(Xs))
        lo = -hi
    else
        lo, hi = extrema(Xs)
    end
    # avoid FFT wrap-around numerical difficulties
    lo_kde, hi_kde = min(a_min, lo - 6 * h), max(a_max, hi + 6 * h)
    midpts = range(lo_kde, hi_kde; length = npoints)

    fitted_kde = kde(Xs, KernelDensity.UniformWeights(m), midpts, kernel)
    interp_kde = InterpKDE(fitted_kde)

    # quantities only at points of interest
    midpts_idx = findall((midpts .>= a_min) .& (midpts .<= a_max))

    if absolute_value
        pos_idx = findall((midpts .> 0) .& (midpts .<= a_max))
        neg_idx  = reverse(findall((midpts .< 0) .& (midpts .>= a_min)))
        midpts[pos_idx] != -midpts[neg_idx] && error("symmetry not satisfied")
        estimated_density = fitted_kde.density[pos_idx] + fitted_kde.density[neg_idx]
    else
        estimated_density = fitted_kde.density[midpts_idx]
    end

    C∞_boot = Vector{Float64}(undef, nboot)

    if bootstrap === :Multinomial
        multi = Multinomial(m, fill(1 / m, m))
    end
    for k = 1:nboot
        if bootstrap === :Poisson # Poisson bootstrap to estimate certainty band
            Z_pois = rand(rng, Poisson(1), m)
            ws = Weights(Z_pois / sum(Z_pois))
        elseif bootstrap === :Multinomial # Efron bootstrap
            ws = Weights(vec(rand(rng, multi, 1)))
        else
            throw(error("Only :Multinomial and :Poisson supported."))
        end
        f_kde_pois = kde(Xs, ws, midpts, kernel)
        if absolute_value
            C∞_boot[k] = maximum(abs.(estimated_density .- f_kde_pois.density[pos_idx] .-  f_kde_pois.density[neg_idx]))
        else
            C∞_boot[k] = maximum(abs.(estimated_density .- f_kde_pois.density[midpts_idx]))
        end
    end

    C∞ = quantile(C∞_boot, 1 - α)

    if absolute_value
        midpoints = fitted_kde.x[pos_idx]
    else
        midpoints = fitted_kde.x[midpts_idx]
    end

    FittedInfinityNormDensityBand(
        C∞ = C∞,
        a_min = a_min,
        a_max = a_max,
        fitted_kde = fitted_kde,
        interp_kde = interp_kde,
        midpoints = midpoints,
        estimated_density = estimated_density,
        boot_samples = C∞_boot,
    )
end


@recipe function f(ctband::FittedInfinityNormDensityBand; subsample = 300)
    y_all = ctband.estimated_density
    x_all = response.(ctband.midpoints)

    n = length(y_all)
    _step = div(n - 2, subsample)

    if _step <= 1
        idxs = 1:1:n
    else
        idxs = [1; 2:_step:(n-1); n]
    end
    x = x_all[idxs]
    y = y_all[idxs]

    a_min = ctband.a_min
    a_max = ctband.a_max

    xlims --> (a_min, a_max)
    ylims --> (0, maximum(y) * 1.1)
    yguide --> "Density"

    α = nominal_alpha(ctband)
    ci_level = 100 * (1 - α)

    cis_ribbon = ctband.C∞
    fillalpha --> 0.36
    seriescolor --> "#018AC4"
    ribbon --> cis_ribbon
    framestyle --> :box
    legend --> :topleft
    label --> "$ci_level% CI"

    x, y
end



function Empirikos.flocalization_constraint!(
    model,
    ctband::FittedInfinityNormDensityBand,
    prior::Empirikos.PriorVariable,
)

    @show

    C∞ = ctband.C∞
    _midpoints = ctband.midpoints
    _density_values = ctband.estimated_density

    if !isa(model, LinearFractional.LinearFractionalModel) && JuMP.solver_name(model) == "Hypatia"
        marginal_pdfs = pdf.(Ref(prior), _midpoints)
        ncone = 1 + length(_midpoints)
        @constraint(model, [C∞; marginal_pdfs .- _density_values] in MathOptInterface.NormInfinityCone(ncone))
    else
        # TODO: perhaps move this evaluation below
        _min, _max = extrema(Empirikos.MarginalDensity(_midpoints[1]))
        for (index, Z) in enumerate(_midpoints)
            _density_hat = _density_values[index]
            marginal_pdf = pdf(prior, Z::EBayesSample)

            if _density_hat + C∞ < _max
                @constraint(model, marginal_pdf <= _density_hat + C∞)
            end
            if _density_hat - C∞ > _min
                @constraint(model, marginal_pdf >= _density_hat - C∞)
            end
        end
    end
    model
end
