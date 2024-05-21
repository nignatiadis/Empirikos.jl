using Test
using Empirikos
using StatsDiscretizations 
using Distributions 

int1 = Interval(-Inf, 0.0)
ebs1 = StandardNormalSample(int1)

@test Empirikos.likelihood(ebs1, 0.0) == 0.5

int2 = Interval{:open, :open}(0.0, Inf)
ebs2 = StandardNormalSample(int2)

@test Empirikos.likelihood(ebs2, 0.0) == 0.5

@code_warntype Empirikos.likelihood(ebs2, 0.0)


vec_heteroskedastic = [NormalSample(0.0, 2.0); NormalSample(0.0, 3.0)]
@test skedasticity(vec_heteroskedastic) == Empirikos.Heteroskedastic()


int = Interval{:open,:closed}(1.0,2.)

interval_normal = NormalSample{typeof(int), Float64}(int, 1.0)
Base.summarysize(interval_normal)
@inferred likelihood(interval_normal, 0.0)


# Test some discrete Interval functionality
tmp = BinomialSample(Interval(0,1), 10)
@test cdf(Binomial(10, 0.5), 1) ≈ likelihood(tmp, 0.5)
@test likelihood(BinomialSample(Interval(0,10), 10), 0.6) == 1


@test pdf(Dirac(0.7), BinomialSample(Interval(9,Inf), 10))  ≈ ccdf(Binomial(10,0.7), 8)
@test pdf(Dirac(0.7), BinomialSample(Interval(9,Inf), 10))  ≈  pdf(Dirac(0.7), BinomialSample(Interval(9,10), 10))


#pdf(DiscreteNonParametric([2.0],[1.0]), PoissonSample(Interval(5,nothing)))
#ccdf(Poisson(2.0), 4)
