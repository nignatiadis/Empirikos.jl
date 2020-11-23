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

# grid could be :default or Integer or actual grid.








#struct InfinityNormBand{T}
#
#end
