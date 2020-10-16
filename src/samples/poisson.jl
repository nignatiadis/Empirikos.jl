struct PoissonSample{T,S} <: EBayesSample{T}
    Z::T
    E::S     # add checks that Z \in {0,...,n}
end

PoissonSample(E) = BinomialSample(missing, E)
PoissonSample() = BinomialSample(missing, 1.0)

response(Z::PoissonSample) = Z.Z
nuisance_parameter(Z::PoissonSample) = Z.E

likelihood_distribution(Z::PoissonSample, λ) = Poisson(λ * nuisance_parameter(E))

function Base.show(io::IO, Z::PoissonSample)
    spaces_to_keep = ismissing(response(Z)) ? 1 : max(3 - ndigits(response(Z)), 1)
    spaces = repeat(" ", spaces_to_keep)
    print(io, "Z=", response(Z), spaces, "| ", "E=", Z.E)
end

# how to break ties on n?
function Base.isless(a::PoissonSample, b::PoissonSample)
    a.E <= b.E && response(a) < response(b)
end
