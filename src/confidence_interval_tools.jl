#
# Confidence Interval Tools

abstract type ConfidenceInterval end

Base.broadcastable(ci::ConfidenceInterval) = Ref(ci)

# LowerUpperConfidenceInterval
Base.@kwdef struct LowerUpperConfidenceInterval <: ConfidenceInterval
    lower::Float64
    upper::Float64
    α::Float64 = 0.05
    estimate::Float64 = (lower + upper)/2
    target = nothing
    method = nothing
end

function Base.show(io::IO, ci::LowerUpperConfidenceInterval)
    print(io, "lower = ", round(ci.lower,sigdigits=4))
    print(io, ", upper = ", round(ci.upper,sigdigits=4))
    print(io, ", α = ", ci.α)
    print(io, "  (", ci.target,")")
end

@recipe function f(bands::AbstractVector{<:ConfidenceInterval})
    x = [Float64(location(band.target)) for band in bands]
    x, bands
end

@recipe function f(x, bands::AbstractVector{<:ConfidenceInterval};  show_ribbon=true)
    lower = getproperty.(bands, :lower)
    upper = getproperty.(bands, :upper)
    estimate = getproperty.(bands, :estimate)

	background_color_legend --> :transparent
	foreground_color_legend --> :transparent
    grid --> nothing
    framestyle --> :box
    legend --> :topleft

    if !show_ribbon
        _label = get(plotattributes, :label, "CI")
        _linestyle = get(plotattributes, :linestyle, :dash)
        _linecolor = get(plotattributes, :linecolor, :black)
        label := [_label nothing]
        linecolor := [_linecolor _linecolor]
        linestyle := [_linestyle _linestyle]
        return x, [lower upper]
    else
	    cis_ribbon  = estimate .- lower, upper .- estimate
	    fillalpha --> 0.36
	    seriescolor --> "#018AC4"
	    ribbon := cis_ribbon
        linealpha --> 0
        return x, estimate
    end
end

function gaussian_ci(se; maxbias=0.0, α=0.05)
    maxbias = abs(maxbias) # should throw an error?
    if iszero(se)
        return maxbias
    end
    rel_bias = maxbias/se
    if abs(rel_bias) > 7
        pm = quantile(Normal(), 1-α) + abs(rel_bias)
    else
        pm = sqrt(quantile(NoncentralChisq(1, abs2(rel_bias)), 1-α))
    end
    se*pm
end

struct BiasVarianceConfidenceInterval <: ConfidenceInterval
    target
    method
    α::Float64
    tail::Symbol
    estimate::Float64
    se::Float64
    maxbias::Float64
    halflength::Float64
    lower::Float64
    upper::Float64
end

function BiasVarianceConfidenceInterval(; target = nothing,
    method = nothing,
    α    = 0.05,
    tail  = :both,
    estimate,
    se,
    maxbias = 0.0)


    if tail === :both
        halflength = gaussian_ci(se; maxbias=maxbias, α=α)
        lower = estimate - halflength
        upper = estimate + halflength
    elseif tail === :right
        halflength = se*quantile(Normal(), 1-α) + abs(maxbias)
        lower = estimate - halflength
        upper = Inf
    elseif tail == :left
        halflength = se*quantile(Normal(), 1-α) + abs(maxbias)
        lower = -Inf
        upper = estimate + halflength
    else
        throw(ArgumentError("tail=$(tail) is not a valid keyword argument"))
    end

    BiasVarianceConfidenceInterval(target,
        method,
        α,
        tail,
        estimate,
        se,
        maxbias,
        halflength,
        lower,
        upper)
end


function Base.show(io::IO, ci::BiasVarianceConfidenceInterval)
    print(io, "lower = ", round(ci.lower,sigdigits=4))
    print(io, ", upper = ", round(ci.upper,sigdigits=4))
    print(io, ", est. = ", round(ci.estimate,sigdigits=4))
    print(io, ", se = ", round(ci.se,sigdigits=4))
    print(io, ", maxbias = ", round(ci.maxbias,sigdigits=4))
    print(io, ", α = ", ci.α)
    print(io, "  (", ci.target,")")
end




Base.@kwdef struct BisectionPair
    var1::Float64
    max_bias1::Float64
    estimate1::Float64
    var2::Float64
    max_bias2::Float64
    estimate2::Float64
    cov::Float64
end

Base.broadcastable(pair::BisectionPair) = Ref(pair)

function confint(pair::BisectionPair, λ ; α=0.05, tail=:both)
    _se = sqrt( abs2(1-λ)*pair.var1 + abs2(λ)*pair.var2 + 2*λ*(1-λ)*pair.cov)
    _maxbias = (1-λ)*pair.max_bias1 + λ*pair.max_bias2
    _estimate = (1-λ)*pair.estimate1 + λ*pair.estimate2

    bw = BiasVarianceConfidenceInterval(;
        α=α, tail=tail, maxbias=_maxbias, se=_se, estimate = _estimate)

    bw.lower, bw.upper
end
