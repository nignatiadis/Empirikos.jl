abstract type EBayesNeighborhood end


Base.@kwdef struct DvoretzkyKieferWolfowitz{T} <: EBayesNeighborhood
    α::T = 0.05
end

function _dkw_level(dkw::DvoretzkyKieferWolfowitz{T}, Zs) where T<:Number
    dkw.α
end

function _dkw_level(dkw::DvoretzkyKieferWolfowitz, Zs)
    dkw.α(nobs(Zs))
end

struct FittedDvoretzkyKieferWolfowitz{T, S, D<:AbstractDict{T,S}}
    summary::D
    band::S
end

function StatsBase.fit(dkw::DvoretzkyKieferWolfowitz, Zs_summary)
    cdf_probs = cumsum([v for (k,v) in Zs_summary.store])
    cdf_probs /= cdf_probs[end]
    _dict = SortedDict( keys(Zs_summary.store) .=> cdf_probs)
    α = _dkw_level(dkw, Zs_summary)
    n = nobs(Zs_summary)
    band = sqrt(log(2/α)/(2n))
    FittedDvoretzkyKieferWolfowitz(_dict, band)
end

 function neighborhood_constraint!(model, dkw::FittedDvoretzkyKieferWolfowitz, prior::PriorVariable)
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
