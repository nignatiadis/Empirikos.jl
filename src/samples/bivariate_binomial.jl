struct BivariateBinomialSample{T,S<:Integer} <: DiscreteEBayesSample{T}
    Z1::BinomialSample{T,S}
    Z2::BinomialSample{T,S}  #TODO: add checks that Z \in {0,...,n}
end

function response(Z::BivariateBinomialSample)
    [response(Z.Z1), response(Z.Z2)]
 end

 function nuisance_parameter(Z::BivariateBinomialSample)
    [nuisance_parameter(Z.Z1), nuisance_parameter(Z.Z2)]
 end

 function set_response(Z::BivariateBinomialSample, znew=missing)
    znew1 = ismissing(znew) ? missing : znew[1]
    znew2 = ismissing(znew) ? missing : znew[2]

    new_Z1 = set_response(Z.Z1, znew1)
    new_Z2 = set_response(Z.Z2, znew2)
    BivariateBinomialSample(new_Z1, new_Z2)
end

function Base.show(io::IO, Z::BivariateBinomialSample)
    resp_Z1 = response(Z.Z1)
    n1 = nuisance_parameter(Z.Z1)
    resp_Z2 = response(Z.Z2)
    n2 = nuisance_parameter(Z.Z2)
   print(io, "ℬ𝒾𝓃(", (resp_Z1, resp_Z2), "; (p₁,p₂), n=", (n1, n2),")")
   # print(io, "ℬ𝒾𝓃(", resp_Z1,"; p₁, n=", n1,")⊗ℬ𝒾𝓃(", resp_Z2,"; p₂, n=", n2,")")
end


function Empirikos.likelihood_distribution(Z::BivariateBinomialSample, p)
    product_distribution(Empirikos.likelihood_distribution(Z.Z1, p[1]), Empirikos.likelihood_distribution(Z.Z2, p[2]))
end

function Empirikos.marginalize(Z::BivariateBinomialSample, prior::Distributions.ProductDistribution)
    product_distribution(
        Empirikos.marginalize(Z.Z1, prior.dists[1]),
        Empirikos.marginalize(Z.Z2, prior.dists[2])
    )
end

function Empirikos.posterior(Z::BivariateBinomialSample, prior::Distributions.ProductDistribution)
    product_distribution(
        Empirikos.posterior(Z.Z1, prior.dists[1]),
        Empirikos.posterior(Z.Z2, prior.dists[2])
    )
end
