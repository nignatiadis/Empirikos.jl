abstract type EBayesNeighborhood end
abstract type FittedEBayesNeighborhood end

# Holy trait
abstract type NeighborhoodVexity end
struct LinearVexity <: NeighborhoodVexity end
struct ConvexVexity <: NeighborhoodVexity end



StatsBase.fit(nbhood::FittedEBayesNeighborhood, args...; kwargs...) = nbhood

function nominal_alpha(nbhood::EBayesNeighborhood)
    nbhood.α
end



function neighborhood_constraint!(model, nbood, prior::PriorVariable)
    model
end

"""
    DvoretzkyKieferWolfowitz(α) <: EBayesNeighborhood

The Dvoretzky-Kiefer-Wolfowitz band (based on the Kolmogorov-Smirnov distance)
at confidence level `1-α` that bounds the distance of the true distribution function
to the ECDF ``\\widehat{F}_n`` based on ``n`` samples. The constant of the band is the sharp
constant derived by Massart:

```math
F \\text{ distribution}:  \\sup_{t \\in \\mathbb R}\\lvert F(t) - \\widehat{F}_n(t) \\rvert  \\leq  \\sqrt{\\log(2/\\alpha)/(2n)}
```

"""
Base.@kwdef struct DvoretzkyKieferWolfowitz{T} <: EBayesNeighborhood
    α::T = 0.05
end

vexity(::DvoretzkyKieferWolfowitz) = LinearVexity()

struct FittedDvoretzkyKieferWolfowitz{T,S,D<:AbstractDict{T,S},DKW} <:
       FittedEBayesNeighborhood
    summary::D
    band::S
    dkw::DKW
end

vexity(dkw::FittedDvoretzkyKieferWolfowitz) = vexity(dkw.dkw)


function nominal_alpha(dkw::FittedDvoretzkyKieferWolfowitz)
    nominal_alpha(dkw.dkw)
end

# TODO: Allow this to work more broadly.
function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs::AbstractVector{<:EBayesSample})
    StatsBase.fit(dkw, summarize(Zs))
end

function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs_summary)
    cdf_probs = cumsum([v for (k, v) in Zs_summary.store])
    cdf_probs /= cdf_probs[end]
    # TODO: report issue with SortedDict upstream
    _dict = SortedDict(Dict(keys(Zs_summary.store) .=> cdf_probs))
    α = nominal_alpha(dkw)
    n = nobs(Zs_summary)
    band = sqrt(log(2 / α) / (2n))
    FittedDvoretzkyKieferWolfowitz(_dict, band, dkw)
end



function neighborhood_constraint!(
    model,
    dkw::FittedDvoretzkyKieferWolfowitz,
    prior::PriorVariable,
)
    band = dkw.band
    for (Z, cdf_value) in dkw.summary
        marginal_cdf = cdf(prior, Z::EBayesSample)
        if cdf_value + band < 1
            @constraint(model, marginal_cdf <= cdf_value + band)
        end
        if cdf_value - band > 0
            @constraint(model, marginal_cdf >= cdf_value - band)
        end
    end
    model
end

@recipe function f(fitted_dkw::FittedDvoretzkyKieferWolfowitz; subsample=100)
    x_dkw_all = response.(collect(keys(fitted_dkw.summary)))
    F_dkw_all = collect(values(fitted_dkw.summary))

    n = length(x_dkw_all)
    step = div(n-2, subsample)
    idxs = [1; 2:step:(n-1); n]

    x_dkw = x_dkw_all[idxs]
    F_dkw = F_dkw_all[idxs]

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
