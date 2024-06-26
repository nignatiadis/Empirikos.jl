"""
   Abstract type representing F-Localizations.
"""
abstract type FLocalization end

"""
   Abstract type representing a fitted F-Localization
   (i.e., wherein the F-localization has already been determined by data).
"""
abstract type FittedFLocalization end

# Holy trait
abstract type FLocalizationVexity end
struct LinearVexity <: FLocalizationVexity end
struct ConvexVexity <: FLocalizationVexity end



StatsBase.fit(floc::FittedFLocalization, args...; kwargs...) = floc

function nominal_alpha(floc::FLocalization)
    floc.α
end



function flocalization_constraint!(model, floc, prior::PriorVariable)
    model
end

"""
    DvoretzkyKieferWolfowitz(;α = 0.05, max_constraints = 1000) <: FLocalization

The Dvoretzky-Kiefer-Wolfowitz band (based on the Kolmogorov-Smirnov distance)
at confidence level `1-α` that bounds the distance of the true distribution function
to the ECDF ``\\widehat{F}_n`` based on ``n`` samples. The constant of the band is the sharp
constant derived by Massart:

```math
F \\text{ distribution}:  \\sup_{t \\in \\mathbb R}\\lvert F(t) - \\widehat{F}_n(t) \\rvert  \\leq  \\sqrt{\\log(2/\\alpha)/(2n)}
```
The supremum above is enforced discretely on at most `max_constraints` number of points.
"""
Base.@kwdef struct DvoretzkyKieferWolfowitz{T,N} <: FLocalization
    α::T = 0.05
    side::Symbol = :both
    max_constraints::N = 1000
end

# for backwards compatibility
DvoretzkyKieferWolfowitz(α) = DvoretzkyKieferWolfowitz(;α=α)

vexity(::DvoretzkyKieferWolfowitz) = LinearVexity()

struct FittedDvoretzkyKieferWolfowitz{T,S,D<:StatsDiscretizations.Dictionary{T,S},DKW} <:
        FittedFLocalization
    summary::D
    band::S
    side::Symbol
    dkw::DKW
    homoskedastic::Bool
end

vexity(dkw::FittedDvoretzkyKieferWolfowitz) = vexity(dkw.dkw)


function nominal_alpha(dkw::FittedDvoretzkyKieferWolfowitz)
    nominal_alpha(dkw.dkw)
end

# TODO: Allow this to work more broadly.
function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs::AbstractVector{<:EBayesSample})
    StatsBase.fit(dkw, summarize(Zs))
end

function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs_summary::MultinomialSummary{T}) where T
    α = nominal_alpha(dkw)
    n = nobs(Zs_summary)
    side = dkw.side

    if skedasticity(Zs_summary) === Heteroskedastic()
        homoskedastic = false
        multiplier = exp(1)
        Zs_summary = compound(Zs_summary)
    elseif skedasticity(Zs_summary) === Homoskedastic()
        if T<:CompoundSample || n != sum(values(Zs_summary))
            homoskedastic = false
            multiplier = exp(1)
        else
            homoskedastic = true
            multiplier = 1
        end
    end
    bonferroni_correction = side == :both ? 2 : 1
    band = sqrt(log(bonferroni_correction * multiplier / α) / (2n))

    n_constraints = length(Zs_summary)


    cdf_probs = cumsum(Zs_summary.store) #SORTED important here
    cdf_probs /= cdf_probs[end]
    _Zs = collect(keys(Zs_summary.store))

    issorted(_Zs, by=response, lt=StatsDiscretizations._isless) || error("MultinomialSummary not sorted.")

    max_constraints = dkw.max_constraints

    if max_constraints < n_constraints - 10
        _step = div(n_constraints-2, max_constraints)
        idxs = [1; 2:_step:(n_constraints-1); n_constraints]
        _Zs = _Zs[idxs]
        cdf_probs = cdf_probs[idxs]
    end
    # TODO: report issue with SortedDict upstream
    _dict = StatsDiscretizations.Dictionary(_Zs, cdf_probs)

    FittedDvoretzkyKieferWolfowitz(_dict, band, side, dkw, homoskedastic)
end



function flocalization_constraint!(
    model,
    dkw::FittedDvoretzkyKieferWolfowitz,
    prior::PriorVariable,
)
    band = dkw.band
    side = dkw.side

    bound_upper = (side == :both) || (side == :upper)
    bound_lower = (side == :both) || (side == :lower)

    for (Z, cdf_value) in zip(keys(dkw.summary), values(dkw.summary))
        marginal_cdf = cdf(prior, Z::EBayesSample)
        if bound_upper
            if cdf_value + band < 1
                @constraint(model, marginal_cdf <= cdf_value + band)
            end
        end
        if bound_lower
            if cdf_value - band > 0
                @constraint(model, marginal_cdf >= cdf_value - band)
            end
        end
    end
    model
end

# add discrete version of this?
@recipe function f(fitted_dkw::FittedDvoretzkyKieferWolfowitz)
    x_dkw = response.(collect(keys(fitted_dkw.summary)))
    F_dkw = collect(values(fitted_dkw.summary))

    band = fitted_dkw.band
    lower = max.(F_dkw .- band, 0.0)
    upper = min.(F_dkw .+ band, 1.0)
    cis_ribbon  = F_dkw .- lower, upper .- F_dkw
    fillalpha --> 0.36
    legend --> :topleft
    seriescolor --> "#018AC4"
    ribbon --> cis_ribbon

    x_dkw, F_dkw
end
