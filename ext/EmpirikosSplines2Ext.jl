module EmpirikosSplines2Ext

using Empirikos
using Splines2
using Statistics

struct ConstantLogVarianceTrend{T}
    value::T
end

(trend::ConstantLogVarianceTrend)(m) = trend.value

struct FittedNaturalSplineVarianceTrend{C,B,K}
    coefficients::C
    boundary_knots::B
    interior_knots::K
    df::Int
    intercept::Bool
end

function Empirikos.fit_trend(
    trend::Empirikos.NaturalSplineVarianceTrend,
    Ms,
    s²s,
)
    isnothing(Ms) &&
        throw(ArgumentError("NaturalSplineVarianceTrend requires passing `Ms` to `fit(test, Zs, Ms)`."))

    length(Ms) == length(s²s) ||
        throw(DimensionMismatch("`Ms` has length $(length(Ms)); expected $(length(s²s))."))

    ms = collect(float.(Ms))
    y = log.(max.(collect(float.(s²s)), trend.min_variance))
    splinedf = _spline_df(trend, ms)

    if splinedf < 2
        return ConstantLogVarianceTrend(mean(y))
    end

    basis = Splines2.ns(ms; df=splinedf, intercept=trend.intercept)
    coefficients = basis \ y
    boundary_knots = extrema(ms)
    interior_knots = _interior_knots(ms, splinedf, trend.intercept)

    FittedNaturalSplineVarianceTrend(
        coefficients,
        boundary_knots,
        interior_knots,
        splinedf,
        trend.intercept,
    )
end

function (trend::FittedNaturalSplineVarianceTrend)(m::Real)
    only(_basis(trend, [float(m)]) * trend.coefficients)
end

function _spline_df(trend::Empirikos.NaturalSplineVarianceTrend, Ms)
    splinedf = if isnothing(trend.df)
        n = length(Ms)
        1 + (n >= 3) + (n >= 6) + (n >= 30)
    else
        trend.df
    end
    min(splinedf, length(unique(Ms)))
end

function _interior_knots(Ms, df, intercept::Bool)
    order = 4
    knots_offset = 2
    nknots = df - order + knots_offset + 1 - Int(intercept)

    if nknots <= 0
        return nothing
    end

    ps = range(0.0, 1.0; length=nknots + 2)[2:(nknots + 1)]
    collect(quantile(Ms, ps))
end

function _basis(trend::FittedNaturalSplineVarianceTrend, Ms)
    Splines2.ns(
        Ms;
        boundary_knots=trend.boundary_knots,
        interior_knots=trend.interior_knots,
        intercept=trend.intercept,
    )
end

end
