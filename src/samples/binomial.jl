struct BinomialSample{T<:Integer, S<:Union{Missing,T}} <: EBayesSample{T}
    Z::S
    n::T     # add checks that Z \in {0,...,n}
end

BinomialSample(n::Integer) = BinomialSample(missing, n)

response(Z::BinomialSample) = Z.Z
ntrials(Z::BinomialSample) = Z.n
nuisance_parameter(Z::BinomialSample) = ntrials(Z)

likelihood_distribution(Z::BinomialSample, p) = Binomial(ntrials(Z), p)

function marginalize(Z::BinomialSample, prior::Beta)
    @unpack α, β =  prior
    BetaBinomial(ntrials(Z), α, β)
end



# Fit BetaBinomial
function StatsBase.fit(::MethodOfMoments{<:Beta}, Zs::AbstractVector{<:BinomialSample}, ::Homoskedastic)
    # TODO: Let ::Homoskedastic carry type information.
    n = ntrials(Zs[1])
    μ₁ = mean(response.(Zs))
    μ₂ = mean(abs2, response.(Zs))
    denom = n*(μ₂/μ₁ - μ₁ -1) + μ₁
    α = (n*μ₁ - μ₂)/denom
    β = (n-μ₁)*(n - μ₂/μ₁)/denom
    Beta(α, β)
end
