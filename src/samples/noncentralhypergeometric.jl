"""
    NonCentralHypergeometricSample(Z, n)
"""
struct NonCentralHypergeometricSample{T} <: DiscreteEBayesSample{T}
    Z::T  #
    m1::T
    m2::T
    n::T
end

function likelihood_distribution(Z::NonCentralHypergeometricSample, θ)
    FisherNoncentralHypergeometric(Z.m1, Z.m2, Z.n, exp(θ))
end


function odds_ratio(Z::NonCentralHypergeometricSample; offset=0.5)
    num = (Z.Z + offset)/(Z.m1 - Z.Z + offset)
    denom = (Z.n - Z.Z + offset)/(Z.m2 - Z.n + Z.Z + offset)
    num/denom
end
