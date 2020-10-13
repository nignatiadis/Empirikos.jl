using Test
using Empirikos

int1 = Interval(nothing, 0.0)
ebs1 = StandardNormalSample(int1)

@test Empirikos.likelihood(ebs1, 0.0) == 0.5

int2 = Interval{Open, Unbounded}(0.0, nothing)
ebs2 = StandardNormalSample(int2)

@test Empirikos.likelihood(ebs2, 0.0) == 0.5

vec_typed = StandardNormalSample.([int1; int1])
vec_not_typed = StandardNormalSample.([int1; int2])
vec_union_typed = StandardNormalSample{EBInterval{Float64}}.([int1;int2])

Empirikos.likelihood.(vec_not_typed, 0.0)

@inferred broadcast(Empirikos.likelihood, vec_typed, 0.0)
#@test_throws @inferred broadcast(Empirikos.likelihood, vec_not_typed, 0.0)
@inferred broadcast(Empirikos.likelihood, vec_union_typed, 0.0)

@test isa(vec_not_typed, AbstractVector{EB} where EB <: EBayesSample)
@test isa(vec_union_typed, AbstractVector{EB} where EB <: EBayesSample)



@code_warntype Empirikos.likelihood(ebs2, 0.0)

@test skedasticity(vec_typed) == Empirikos.Homoskedastic()
@test skedasticity(vec_not_typed) == Empirikos.Homoskedastic()

vec_heteroskedastic = [NormalSample(0.0, 2.0); NormalSample(0.0, 3.0)]
@test skedasticity(vec_heteroskedastic) == Empirikos.Heteroskedastic()
