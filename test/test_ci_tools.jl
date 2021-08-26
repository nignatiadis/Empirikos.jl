using Empirikos
using Distributions
using Test
using Roots

biases = 0:0.1:2.0
standard_errors = 0.1:0.1:3.0
αs = 0.01:0.01:0.2


function gaussian_ci2(se; maxbias=0.0, α=0.05)
    level = 1 - α
    maxbias = abs(maxbias)
    rel_bias = maxbias/se
    zz = fzero( z-> cdf(Normal(), rel_bias-z) + cdf(Normal(), -rel_bias-z) +  level -1,
        0, rel_bias - quantile(Normal(),(1- level)/2.1))
    zz*se
end

for bias in biases, se in standard_errors, α in αs
    ci1 = Empirikos.gaussian_ci(se; maxbias=bias, α=α)
    ci2 = gaussian_ci2(se; maxbias=bias, α=α)
    @test ci1 ≈ ci2
end

for se in standard_errors, α in αs
    ci1 = Empirikos.gaussian_ci(se; α=α)
    ci2 = Empirikos.gaussian_ci(se; maxbias =0.0, α=α)
    ci3 = quantile(Normal(), 1-α/2)*se
    @test ci1 ≈ ci2
    @test ci1 ≈ ci3
end

ci = Empirikos.BiasVarianceConfidenceInterval(estimate=1.0, se=1.0)
@test ci.lower ≈ 1 - 1.96 atol = 0.001
@test ci.upper ≈ 1 + 1.96 atol = 0.001

ci_no_se = Empirikos.BiasVarianceConfidenceInterval(estimate=1.0, se=0.0, maxbias=1.0)
@test ci_no_se.lower ≈ 1.0 - 1.0
@test ci_no_se.upper ≈ 1.0 + 1.0

@test Empirikos.gaussian_ci(0.0; maxbias=2.0) == 2.0
