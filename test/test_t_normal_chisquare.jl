using Empirikos
using Test
using Distributions
using QuadGK
using Random

ν = 3
n = ν+1
Z = NormalChiSquareSample(5.0, abs2(2.0), ν)
T = NoncentralTSample(Z)

# Check constructor 
Random.seed!(1)
Zs = rand(Normal(), n )
Zbar_scaled = sqrt(n) * mean(Zs)
MeanSq = mean(abs2, Zs)
σ̂² = var(Zs; corrected=true)
Zsample =  NormalChiSquareSample(Zbar_scaled, σ̂², ν)
@test Zsample.mean_squares ≈ MeanSq
@test Zsample.mean_squares_dof == n 
@test Zsample.S² == σ̂²
@test Zsample.mean_squares_dof * Zsample.mean_squares ≈ (Zsample.ν + abs2(Zsample.tstat))*Zsample.S²

@test likelihood(Z, (λ=1.0, σ²=4.0)) ≈ likelihood(Z, (μ=2.0, σ²=4.0))
@test pdf(product_distribution((λ=Dirac(1.0), σ²=Dirac(4.0))), Z)  ≈ likelihood(Z, (λ=1.0, σ²=4.0)) 
@test pdf(product_distribution((λ=Dirac(1.0), σ²=DiscreteNonParametric([4.0],[1.0]))), Z)  ≈ likelihood(Z, (λ=1.0, σ²=4.0)) 

@test marginalize(Z, product_distribution((μ=Normal(1.0, 3.0), σ²=Dirac(4.0)))).dists[1] == Normal(1.0, sqrt(4.0 + 9))

# only implemented for centered normal so far
@test_throws Exception marginalize(tmp, Normal(0.5))

prior = Normal(0.0, 2.0) 
@test quadgk(λ-> likelihood(T,  λ)*pdf(prior, λ), -Inf, Inf)[1] ≈ pdf(prior, T)


# tests regarding evalues 

num_normal = Normal(0.0, 2.0)
eval1 = Empirikos.MixtureEValue(num_normal, Dirac(0.0))



# compute analytically, Wang and Ramdas (2025)
function hongjian_ttest_evalue(T, ν, c²) # t statistic, ν dof, c² prior precision
    n = ν + 1
    term1 = log(c² / (n + c²))/2
    _denom = (n + c²) * (n - 1) / abs2(T)
    term2 = log1p(n / (_denom + c²))*(n/2)
    result = exp(term1 + term2)
    return result
end
# remark: In Wang and Ramdas, things are parameterized in terms of the prior precision c²
# of μ. However above we rescaled things by \sqrt{n} \mu$, so we need to adjust accordingly.
c² = n / var(num_normal)

@test eval1(T) == hongjian_ttest_evalue(T.Z, T.ν, c²)

# check universal e-value against Hongjian's computation 
# (Here for sanity check, and we just plug-in fixed values for μ, \sigma²
# and do not update them stepwise
function hongjian_universal_evalue(Zs, μ, σ²) # \mu, \sigma^2 fixed parameters under alternative
    n = length(Zs)
    meansq = mean(abs2, Zs)
    σ = sqrt(σ²)
    log_eval = n/2*(log(meansq) + 1) - sum( log(σ) .+ abs2.( (Zs .- μ)./σ)/2)
    exp(log_eval)
end 

hongjian_universal_evalue(Zs, 2.3, 4.11)
eval2 = Empirikos.MixtureUniversalEValue(product_distribution((μ=Dirac(sqrt(n)*2.3), σ²=Dirac(4.11))))
@test eval2(Zsample) ≈ hongjian_universal_evalue(Zs, 2.3, 4.11)



S_squared = ScaledChiSquareSample( (ν + abs2(response(T))) * Z.S² / (ν+1), ν+1)

σ = 2.0
q = Empirikos.InverseScaledChiSquare(4.0, 5)


function logmyeval(t, s²; ν)
    n = ν+1
    t² = abs2(t)
    T = NoncentralTSample(t, ν)
    σ̂² =  ScaledChiSquareSample(  n* s²/ (ν + t²),  ν)
    S² = ScaledChiSquareSample(s², n)

    q = Empirikos.InverseScaledChiSquare(4.0, 5)
    prior = Normal(0.0, 2.0) 
    var_dbn_null = Dirac(1.0)

    logpdf(prior, T) - logpdf(Dirac(0), T) + log(n/(ν + t²)) + logpdf(q, σ̂²) - logpdf(var_dbn_null, S²)
end

