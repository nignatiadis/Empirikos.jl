lfsr_0 = Empirikos.PosteriorProbability(StandardNormalSample(0.0), Interval(0,nothing))

lfsr_numerator = numerator(lfsr_0)
lfsr_denominator = denominator(lfsr_0)

_num = lfsr_numerator(Normal())
_denom = lfsr_denominator(Normal())
@test _num / _denom == 0.5



prior = Normal()
target = PosteriorMean(StandardNormalSample(2.0))

@test target(MixtureModel([prior; prior], [0.3;0.7])) ≈ target(prior)
