using SpecialFunctions
using Empirikos
using QuadGK
using Setfield
using Test 
using Random


function _limma_pvalue_tweedie(β_hat, Z::ScaledChiSquareSample, prior)
    ν_lik = Z.ν
    s² = response(Z)
    function C(ν)
        num = (1+1/ν)^(-ν/2)*gamma((ν+1)/2)*sqrt(ν +1)
        denom = sqrt(pi)*gamma(ν/2)
        num/denom
    end
    function fp1(u)
        S = ScaledChiSquareSample(u, ν_lik+1)
        pdf(prior, S)
    end
    function integrand(u)
        u^(-(ν_lik-1)/2) * fp1(u) / sqrt((ν_lik+1)*u - ν_lik*s²)
    end
    lower_end = (ν_lik*s² + abs2(β_hat))/(ν_lik+1)
    term1 = (s²)^(ν_lik/2-1)*C(ν_lik)/ pdf(prior, Z)
    term2 = quadgk(integrand, lower_end, Inf)[1]
    term1 * term2
end

Gsingle = DiscreteNonParametric([1.0], [1.0])
Glimma = Empirikos.InverseScaledChiSquare(3.1, 2.0)
Gmix = DiscreteNonParametric([1.0;0.5], [1/2; 1/2])

νs = [1;2;2.5; 3; 10; 31]
βs_hat= [-2.0; 0.01; 0.5; 2.0; 4.0]
s = [0.2 1.0 4.0]
Zs = ScaledChiSquareSample.(s, νs)

Gs = [Gsingle, Glimma, Gmix]
for β in βs_hat, Z in Zs, G in Gs 
    @show β, Z 
    @test (@show _limma_pvalue_tweedie(β, Z, G)) ≈ (@show Empirikos.limma_pvalue(β, Z, G)) 
end

# Check chi square likelihood 

s²s = [0.001; 1.0; 3.0; 10.0; 100.0]
σ²s = [0.001; 2.0; 4.0; 10.0; 500.0]
νs = [2; 3; 5.5; 10; 100]

function scaled_chisq_lik(s², σ², ν)
    (ν/2)^(ν/2) / gamma(ν/2) / σ²^(ν/2) * s²^(ν/2-1) * exp(-ν*s²/(2*σ²))
end

for s² in s²s, σ² in σ²s, ν in νs
    @test scaled_chisq_lik(s², σ², ν) ≈ likelihood(ScaledChiSquareSample(s², ν), σ²)
end

# Test NPMLE 

## Note that the below was the reason to use rescaled likelihoods in NPMLE.jl
Ss = ScaledChiSquareSample.([0.074; 0.5; 2.0; 2.0; 2.0], 64)
priorclass = DiscretePriorClass(0.53:0.01:2.0)
_hyp_fit = fit(NPMLE(priorclass, Hypatia.Optimizer), Ss)

# test against mosek
@test Empirikos.loglikelihood(Ss, _hyp_fit.prior) ≈ -33.614 atol=1e-3


# TODO: also add some more tests, also e.g. on Limma.




