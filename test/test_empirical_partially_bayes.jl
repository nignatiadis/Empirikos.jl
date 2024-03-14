using SpecialFunctions
using Empirikos
using QuadGK
using Setfield
using Test 
using Random
Empirikos.limma_pvalue(β_hat, Z::ScaledChiSquareSample, prior::MixtureModel)


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

# also add some more tests, also e.g. on Limma.




