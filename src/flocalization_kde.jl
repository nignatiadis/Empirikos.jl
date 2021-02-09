abstract type InfiniteOrderKernel <: ContinuousUnivariateDistribution end

function Base.show(io::IO, d::InfiniteOrderKernel)
    print(io, Base.typename(typeof(d)))
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
SincKernel() = SincKernel(DataBasedDefault())

function Empirikos._set_defaults(kernel::InfiniteOrderKernel, Zs::AbstractVector{<:Empirikos.AbstractNormalSample}; kwargs...)
    Empirikos.skedasticity(Zs) == Empirikos.Homoskedastic() || throw("Only implemented for Homoskedastic Gaussian data.")
    σ = std(Zs[1])
    n = length(Zs)
    h = σ/sqrt(log(n))
    @set kernel.h = h
end


Distributions.cf(a::SincKernel, t) = one(Float64)*(-1/a.h <= t <= 1/a.h)

function Distributions.pdf(a::SincKernel, t)
   if t==zero(Float64)
       return(one(Float64)/pi/a.h)
   else
       return(sin(t/a.h)/pi/t)
   end
end

# TODO, General FlatTopKernel

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

DeLaValleePoussinKernel() = DeLaValleePoussinKernel(DataBasedDefault())

function Distributions.cf(a::DeLaValleePoussinKernel, t)
   if abs(t * a.h) <= 1
       return(one(Float64))
   elseif abs(t * a.h) <=  2
       return(2*one(Float64) - abs(t * a.h))
   else
       return(zero(Float64))
   end
end

function Distributions.pdf(a::DeLaValleePoussinKernel, t)
   if t==zero(Float64)
       return(3*one(Float64)/2/pi/a.h)
   else
       return(a.h*(cos(t/a.h)-cos(2*t/a.h))/pi/t^2)
   end
end



"""
    FlatTopKernel(h) < InfiniteOrderKernel

Implements the `FlatTopKernel` with bandwidth `h` to be used for kernel density estimation
through the `KernelDensity.jl` package. The flat-top kernel is defined as follows:
```math
K(x) = \\frac{\\sin^2(1.1x/2)-\\sin^2(\\zo/2)}{\\pi x^2/ 20},
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
"""

struct FlatTopKernel{H} <: InfiniteOrderKernel
    h::H
end

 FlatTopKernel() = FlatTopKernel(DataBasedDefault())

function Distributions.cf(a::FlatTopKernel, t)
    if abs(t * a.h) <= 1
        return(one(Float64))
    elseif abs(t * a.h) <=  1.1
        return(one(Float64) - 10*(abs(t * a.h) - one(Float64)))
    else
        return(zero(Float64))
    end
 end

function Distributions.pdf(a::FlatTopKernel, t)
    if t==zero(Float64)
        return((0.55^2 - 0.5^2)*20/pi)
    else
        return((abs2(sin(0.55*x)) - abs2(sin(0.5*x)))/pi/x^2 * 20)
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
* Second, a `bootstrap` (options: `:Multinomial` or `Poisson`) with `nboot` bootstrap replicates
will be used to estimate ``c_n``, such that:
```math
\\liminf_{n \\to \\infty}\\mathbb{P}[\\sup_{x \\in [a_{\\text{min}} , a_{\\text{max}}]} | \\bar{f}(x) - f(x)| \\leq c_ n} \\geq 1-\\alpha
```
Note that the bound is valid from `a_min` to `a_max`. `alpha` is the nominal level and finally
`rng` sets the seed for the bootstrap samples.
"""
Base.@kwdef struct InfinityNormDensityBand <: FLocalization
   a_min = DataBasedDefault()
   a_max = DataBasedDefault()
   npoints::Integer = 1024
   kernel = FlatTopKernel()
   bootstrap = :Multinomial
   nboot::Integer = 1000
   α::Float64 = 0.05
   rng = Random.MersenneTwister(1)
end

Empirikos.vexity(::InfinityNormDensityBand) = Empirikos.LinearVexity()
function Empirikos._set_defaults(method::InfinityNormDensityBand,
                                Zs::AbstractVector{<:Empirikos.AbstractNormalSample{<:Number}};
                                hints...)
    Empirikos.skedasticity(Zs) == Empirikos.Homoskedastic() || throw("Only implemented for Homoskedastic Gaussian data.")

    q = get(hints, :quantile, 0.1)
    a_min, a_max = quantile(response.(Zs),  (q, 1-q))
    if isa(method.a_min, DataBasedDefault)
        method = @set method.a_min = a_min
    end
    if isa(method.a_max, DataBasedDefault)
        method = @set method.a_max = a_max
    end
    method
end

"""
    FittedInfinityNormDensityBand

The result of running
```julia
StatsBase.fit(opt::InfinityNormDensityBand, Zs)```
Here `opt` is an instance
of [`InfinityNormDensityBand`](@ref) and `Zs` is a vector of [`AbstractNormalSample`](@ref)s
distributed according to a density ``f``..

## Fields:
* `a_min`,`a_max`, `kernel`: These are the same as the fields in `opt::InfinityNormDensityBand`.
* `C∞`
* `fitted_kde`: The fitted `KernelDensity` object.
"""
Base.@kwdef struct FittedInfinityNormDensityBand{T<:Real, S, K} <: FittedFLocalization
    C∞::T
    a_min::T
    a_max::T
    fitted_kde
    interp_kde
    midpoints::S
    estimated_density::K
    boot_samples
    method = nothing
end

Empirikos.vexity(method::FittedInfinityNormDensityBand) = Empirikos.LinearVexity()

function Empirikos.nominal_alpha(inftyband::FittedInfinityNormDensityBand)
    Empirikos.nominal_alpha(inftyband.method)
end

function StatsBase.fit(opt::InfinityNormDensityBand,
                       Zs::AbstractVector{<:Empirikos.AbstractNormalSample{<:Number}}; kwargs...)
    Empirikos.skedasticity(Zs) == Empirikos.Homoskedastic() || throw("Only implemented for Homoskedastic Gaussian data.")
    opt = Empirikos.set_defaults(opt, Zs; kwargs...)

    @unpack a_min, a_max, npoints, kernel, nboot, α, bootstrap, rng = opt

    res = certainty_banded_KDE(response.(Zs), a_min, a_max;
                         npoints = npoints,
                         rng = rng,
                         kernel = kernel,
                         bootstrap = bootstrap,
                         nboot = nboot, α=α)
    res = @set res.method = opt
    Z = Zs[1]
    normal_midpoints =  [@set Z.Z = mdpt for mdpt in res.midpoints]
    @set res.midpoints = normal_midpoints
end

function certainty_banded_KDE(Xs, a_min, a_max;
                        kernel,
                        rng = Random._GLOBAL_RNG,
                        npoints = 4096,
                        nboot = 1_000,
                        bootstrap = :Multinomial,
                        α=0.5)

    h = kernel.h
    m = length(Xs)

    lo, hi = extrema(Xs)
    # avoid FFT wrap-around numerical difficulties
    lo_kde, hi_kde = min(a_min, lo - 6*h), max(a_max, hi + 6*h)
    midpts = range(lo_kde, hi_kde; length = npoints)

    fitted_kde = kde(Xs, KernelDensity.UniformWeights(m), midpts, kernel)
    interp_kde = InterpKDE(fitted_kde)
    midpts_idx = findall( (midpts .>= a_min) .& (midpts .<= a_max) )
    C∞_boot = Vector{Float64}(undef, nboot)

    if bootstrap === :Multinomial
        multi = Multinomial(m, fill(1/m, m))
    end
    for k =1:nboot
        # Poisson bootstrap to estimate certainty band
        if bootstrap === :Poisson
            Z_pois = rand(rng, Poisson(1), m)
            ws =  Weights(Z_pois/sum(Z_pois))
        elseif bootstrap === :Multinomial
            ws = Weights(vec(rand(rng, multi, 1)))
        else
            throw(error("Only :Multinomial and :Poisson supported."))
        end
        f_kde_pois =  kde(Xs, ws, midpts, kernel)
        C∞_boot[k] = maximum(abs.(fitted_kde.density[midpts_idx] .- f_kde_pois.density[midpts_idx]))
    end

    C∞ = quantile(C∞_boot, 1-α)

    FittedInfinityNormDensityBand(C∞=C∞,
                    a_min=a_min,
                    a_max=a_max,
                    fitted_kde=fitted_kde,
                    interp_kde=interp_kde,
                    midpoints = fitted_kde.x[midpts_idx],
                    estimated_density = fitted_kde.density[midpts_idx],
                    boot_samples = C∞_boot)
end


@recipe function f(ctband::FittedInfinityNormDensityBand; subsample=300)
    y_all = ctband.fitted_kde.density
    x_all = ctband.fitted_kde.x
    a_min = ctband.a_min
    a_max = ctband.a_max

    _in_amin_amax = a_min .<= x_all .<= a_max
    x_all = x_all[_in_amin_amax]
    y_all = y_all[_in_amin_amax]

    n = length(y_all)
    _step = div(n-2, subsample)
    idxs = [1; 2:_step:(n-1); n]
    x = x_all[idxs]
    y = y_all[idxs]


    xlims --> (a_min, a_max)
    ylims --> (0, maximum(y)*1.1)
    yguide --> "Density"

    α = nominal_alpha(ctband)
    ci_level = 100*(1-α)

	cis_ribbon  = ctband.C∞
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
    prior::Empirikos.PriorVariable)


    C∞ = ctband.C∞
    _midpoints = ctband.midpoints
    _density_values = ctband.estimated_density

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
    model
end
