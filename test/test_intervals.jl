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

int3 = Interval{Closed, Closed}(1.0,2.0)
eb3 = NormalSample{typeof(int3),Float64}(int3, 1.0)
@inferred Empirikos.likelihood(eb3, 0.0)

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



int = Interval{Open,Closed}(1.0,2.)

interval_normal = NormalSample{typeof(int), Float64}(int, 1.0)
Base.summarysize(interval_normal)
@inferred likelihood(interval_normal, 0.0)

# What happens with union type?
union_interval_normal = NormalSample(int, 1.0)
Base.summarysize(union_interval_normal)
@inferred likelihood(union_interval_normal, 0.0)

any_normal = NormalSample{Any, Float64}(int, 1.0);
Base.summarysize(any_normal)
#@inferred likelihood(any_normal, 0.0)

ints = [Interval{Open,Closed}(1.0,2.0);Interval{Unbounded,Closed}(nothing,2.0);Interval{Open,Unbounded}(1.0,nothing)]
multinormals = NormalSample.(ints, 2.0)
untyped_multinormals = [NormalSample{typeof(int), Float64}(int, 2.0) for int in ints]
Base.summarysize(multinormals)
Base.summarysize(untyped_multinormals)



# Test some discrete Interval functionality
tmp = BinomialSample(Interval(0,1), 10)
cdf(Binomial(10, 0.5), 1) == likelihood(tmp, 0.5)

#pdf(DiscreteNonParametric([2.0],[1.0]), PoissonSample(Interval(5,nothing)))
#ccdf(Poisson(2.0), 4)
