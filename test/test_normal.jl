using Empirikos
using Test


_target = MarginalDensity(StandardNormalSample(3.0))
@test Base.extrema(_target)[2] == pdf(Normal(),0)

@test Base.extrema(MarginalDensity(NormalSample(3.0, 0.5)))[2] == pdf(Normal(0,0.5),0.0)
