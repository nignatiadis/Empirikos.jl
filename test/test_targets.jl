using Empirikos
using QuadGK
using ForwardDiff
using FiniteDifferences
prior = Normal()

@test Empirikos._support(Normal()) == Interval(-Inf, +Inf)

lfsr_0 = Empirikos.PosteriorProbability(StandardNormalSample(0.0), Interval(0, Inf))

@test lfsr_0(-1.0) == 0.0
@test lfsr_0(0.1) == 1.0

lfsr_numerator = numerator(lfsr_0)
lfsr_denominator = denominator(lfsr_0)

@test Empirikos._support(lfsr_numerator) == Interval(0, Inf)

_num = lfsr_numerator(prior)
_denom = lfsr_denominator(prior)
@test _num / _denom == 0.5

@test Empirikos.default_target_computation(lfsr_numerator, location(lfsr_numerator), prior) == Empirikos.NumeratorOfConjugate()
@test Empirikos.compute_target(Empirikos.NumeratorOfConjugate(), lfsr_numerator, prior) == _num

@test Empirikos.compute_target(Empirikos.QuadgkQuadrature(), lfsr_numerator, prior) ≈ _num
@test Empirikos.compute_target(Empirikos.QuadgkQuadrature(), lfsr_denominator, prior) ≈ _denom

@test numerator(lfsr_0)(Uniform(-1.0,0.0)) == 0.0

postmean_target = PosteriorMean(StandardNormalSample(2.0))
postmean_target_binomial = PosteriorMean(BinomialSample(3,10))

@test postmean_target(MixtureModel([prior; prior], [0.3;0.7])) ≈ postmean_target(prior)

@test postmean_target(2.1) == 2.1
@test postmean_target_binomial(0.6) == 0.6

num_target_gaussian = Empirikos.PosteriorTargetNumerator(postmean_target)
num_target_gaussian(1.0)

num_target_binomial = Empirikos.PosteriorTargetNumerator(postmean_target_binomial)
num_target_binomial(Beta(2,1))
num_target_binomial(0.4)

#Empirikos.PosteriorTargetNumerator{PosteriorMean{BinomialSample{Int64,Int64}}})

@test Empirikos.default_target_computation(num_target_gaussian, location(num_target_gaussian), prior) == Empirikos.NumeratorOfConjugate()
@test Empirikos.compute_target(Empirikos.NumeratorOfConjugate(), num_target_gaussian, prior) == num_target_gaussian(prior)

Empirikos.compute_target(Empirikos.QuadgkQuadrature(), lfsr_numerator, prior)

Z = location(num_target_gaussian)

@test quadgk(μ-> likelihood(Z, μ)*μ*pdf(prior,μ), -Inf, +Inf)[1] ≈  num_target_gaussian(prior)
@test quadgk(μ-> likelihood(Z, μ)*μ*pdf(prior,μ), -Inf, +Inf)[1] ==
     Empirikos.compute_target(Empirikos.QuadgkQuadrature(), numerator(postmean_target), prior)

_f_num_postmean_gaussian_quad(μ) = Empirikos.compute_target(Empirikos.QuadgkQuadrature(), numerator(postmean_target), Normal(μ))
@test ForwardDiff.derivative(_f_num_postmean_gaussian_quad, 0.0) ≈ FiniteDifferences.central_fdm(5,1)(_f_num_postmean_gaussian_quad, 0.0)
@test ForwardDiff.derivative(_f_num_postmean_gaussian_quad, -2.0) ≈ FiniteDifferences.central_fdm(5,1)(_f_num_postmean_gaussian_quad, -2.0)

_f_postmean_gaussian(μ) = postmean_target(Normal(μ))
@test ForwardDiff.derivative(_f_postmean_gaussian, 0.0) ≈ FiniteDifferences.central_fdm(5,1)(_f_postmean_gaussian, 0.0)
@test ForwardDiff.derivative(_f_postmean_gaussian, 2.0) ≈ FiniteDifferences.central_fdm(5,1)(_f_postmean_gaussian, 2.0)

#_f_lfsr3(μ::T) where {T} = lfsr_0(Normal{T}(μ), one(T))
#@test ForwardDiff.derivative(_f_lfsr3, 0.0) ≈ FiniteDifferences.central_fdm(5,1)(_f_lfsr, 0.0)
#@test ForwardDiff.derivative(_f_lfsr, 2.0) ≈ FiniteDifferences.central_fdm(5,1)(_f_lfsr, 2.0)
