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

# Test 1: SymmetricPosteriorMean with Asymmetric prior
prior = Normal(1.0, 0.5) 
Z = NormalSample(0.8, 1.0)
    
sym_prior = MixtureModel([Normal(1.0, 0.5), Normal(-1.0, 0.5)], [0.5, 0.5])
    
expected = Empirikos.compute_target(Empirikos.Conjugate(), PosteriorMean(Z), Z, sym_prior)
    
computed = Empirikos.compute_target(Empirikos.Conjugate(), Empirikos.SymmetrizedPosteriorMean(Z), Z, prior)
    
posterior_mean1 = (1.0/(0.5^2) + 0.8/(1.0^2)) / (1/(0.5^2) + 1/(1.0^2))  

posterior_mean2 = (-1.0/(0.5^2) + 0.8/(1.0^2)) / (1/(0.5^2) + 1/(1.0^2))  

marginal_z1 = pdf(Normal(1.0, √(0.5^2 + 1.0^2)), 0.8)  
marginal_z2 = pdf(Normal(-1.0, √(0.5^2 + 1.0^2)), 0.8)

w1 = 0.5 * marginal_z1
w2 = 0.5 * marginal_z2
total_weight = w1 + w2
weight1 = w1 / total_weight
weight2 = w2 / total_weight
theoretical_value = weight1 * posterior_mean1 + weight2 * posterior_mean2
@test isapprox(expected, computed, atol=1e-6)
@test isapprox(computed, theoretical_value, atol=1e-6)
@test isapprox(expected, theoretical_value, atol=1e-6)
    
# Test 2: SymmetricPosteriorMean with Symmetric prior
prior = Normal(0.0, 1.0)
Z = NormalSample(2.0, 1.0)
    
theoretical = (0.0/1.0 + 2.0/1.0) / (1/1.0 + 1/1.0)

computed = Empirikos.compute_target(Empirikos.Conjugate(), Empirikos.SymmetrizedPosteriorMean(Z), Z, prior)
expected = Empirikos.compute_target(Empirikos.Conjugate(), PosteriorMean(Z), Z, prior)
    
@test isapprox(theoretical, computed, atol=1e-6)
@test isapprox(expected, computed, atol=1e-6)


#Test for SignAgreementProbability
function signagree_closed_form(z, μ, τ)
    σ = sqrt(1 + τ^2)
    s     = τ / σ
    m_plus  = (τ^2 * z + μ) / (1 + τ^2)
    m_minus = (-τ^2 * z + μ) / (1 + τ^2)

    fz   = pdf(Normal(μ, σ),  z)
    fneg = pdf(Normal(μ, σ), -z)

    num = fz   * cdf(Normal(),  m_plus / s) +
          fneg * cdf(Normal(), -m_minus / s)
    den = fz + fneg
    return num / den
end

function signagree_closed_form_normalsample(z, μ, τ)
    σ = sqrt(1 + τ^2)
    s = τ / σ
    m = (τ^2 * z + μ) / (1 + τ^2)

    if z >= 0
        return cdf(Normal(),  m / s)    
    else
        return cdf(Normal(), -m / s)   
    end
end
@testset "SignAgreementProbability" begin
    #test for numerator functionality (folded normal)
    t   = Empirikos.SignAgreementProbability(FoldedNormalSample(4))
    t_num = numerator(t)
    p = Normal(1,2)
    @test t_num(p) == numerator(Empirikos.PosteriorProbability(StandardNormalSample(4), Interval{:open,:open}(0.0, Inf)))(p) +
                     numerator(Empirikos.PosteriorProbability(StandardNormalSample(-4), Interval{:open,:open}(-Inf, 0.0)))(p)
    #test for normal prior (folded normal)
    for (μ, τ) in ((0.0, 0.5), (0.0, 1.0), (0.4, 0.7), (-0.8, 1.2), (1.0, 2.0))
        prior = Normal(μ, τ)
        for z in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
            t    = Empirikos.SignAgreementProbability(FoldedNormalSample(abs(z)))
            num  = numerator(t)(prior)
            den  = denominator(t)(prior)
            val  = t(prior)
            val_impl  = num / den
            @test isapprox(val, val_impl; rtol=1e-11, atol=1e-12)
            val_exact = signagree_closed_form(abs(z), μ, τ)
            @test isapprox(val_impl, val_exact; rtol=1e-11, atol=1e-12)
        end
    end
    #test for uniform prior (folded normal)
    prior = Uniform(-10.0, 10.0)
    t   = Empirikos.SignAgreementProbability(FoldedNormalSample(4))
            
    num_pos, _ = quadgk(μ -> (μ > 0) * pdf(Normal(μ, 1), 4) * pdf(prior, μ), -10, 10)

    num_neg, _ = quadgk(μ -> (μ < 0) * pdf(Normal(μ, 1), -4) * pdf(prior, μ), -10, 10)
    
    @test isapprox(numerator(t)(prior), num_pos + num_neg, atol=1e-6)

    #test for numerator functionality (normal sample)
    prior = Normal(1, 2)
    Zp = NormalSample(4.0, 1)
    tp = Empirikos.SignAgreementProbability(Zp)
    @test numerator(tp)(prior) ==
          numerator(Empirikos.PosteriorProbability(Zp, Interval{:open,:open}(0.0, Inf)))(prior)

    Zn = NormalSample(-4.0, 1)
    tn = Empirikos.SignAgreementProbability(Zn)
    @test numerator(tn)(prior) ==
          numerator(Empirikos.PosteriorProbability(Zn, Interval{:open,:open}(-Inf, 0.0)))(prior)

    #test for Normal prior (Normal sample)
    for (μ, τ) in ((0.0, 0.5), (0.0, 1.0), (0.4, 0.7), (-0.8, 1.2), (1.0, 2.0))
        prior = Normal(μ, τ)
        for z in (-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0)
            t = Empirikos.SignAgreementProbability(NormalSample(z, 1))
            num = numerator(t)(prior)
            den = denominator(t)(prior)
            val  = t(prior)
            val_impl  = num / den
            @test isapprox(val, val_impl; rtol=1e-11, atol=1e-12)
            val_exact = signagree_closed_form_normalsample(z, μ, τ)
            @test isapprox(val, val_exact; rtol=1e-11, atol=1e-12)
        end
    end

    #test for uniform prior (Normal sample)
    prior = Uniform(-10.0, 10.0)

    for z in (4.0, -4.0)
        t = Empirikos.SignAgreementProbability(NormalSample(z, 1.0))

        if z > 0
            num_quad, _ = quadgk(μ -> (μ > 0) * pdf(Normal(μ, 1), z) * pdf(prior, μ), -10, 10)
        else
            num_quad, _ = quadgk(μ -> (μ < 0) * pdf(Normal(μ, 1), z) * pdf(prior, μ), -10, 10)
        end

        @test isapprox(numerator(t)(prior), num_quad; atol=1e-6)
    end

end

#Test for ReplicationProbability
function repl_closed_form(z, μ, τ)
    σ = sqrt(1 + τ^2 / (1 + τ^2))
    mplus  = (τ^2 * z + μ) / (1 + τ^2)
    mminus = (-τ^2 * z + μ) / (1 + τ^2)
    σm = sqrt(1 + τ^2)
    fz   = pdf(Normal(μ, σm),  z)
    fneg = pdf(Normal(μ, σm), -z)

    prob_plus  = ccdf(Normal(mplus,  σ),  1.96)
    prob_minus =  cdf(Normal(mminus, σ), -1.96)

    num = fz * prob_plus + fneg * prob_minus
    den = fz + fneg
    return num / den
end

function repl_closed_form_normalsample(z, μ, τ)
    σ = sqrt(1 + τ^2 / (1 + τ^2))    
    m = (τ^2 * z + μ) / (1 + τ^2)

    if z >= 0
        return ccdf(Normal(m,  σ),  1.96)    
    else
        return cdf(Normal(m, σ), -1.96)
    end
end


@testset "ReplicationProbability" begin
    #test for normal prior (folded normal)
    for (μ0, τ) in ((0.0, 0.5), (0.0, 1.0), (0.4, 0.7), (-0.8, 1.2), (1.0, 2.0))
        prior = Normal(μ0, τ)
        for z in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
            t = Empirikos.ReplicationProbability(FoldedNormalSample(abs(z)))
            val = t(prior)
            val_impl = numerator(t)(prior) / denominator(t)(prior)
            @test isapprox(val, val_impl; rtol=1e-11, atol=1e-12)

            val_exact = repl_closed_form(abs(z), μ0, τ)
            @test isapprox(val, val_exact; rtol=1e-10, atol=1e-12)
        end
    end
    
    #test for uniform prior (folded normal)
    prior = Uniform(-10.0, 10.0)
    z = 4.0
    t = Empirikos.ReplicationProbability(FoldedNormalSample(z))

    term_plus, _ = quadgk(μ -> pdf(Normal(μ, 1),  z) * ccdf(Normal(μ, 1),  1.96) * pdf(prior, μ), -10, 10)
    term_minus,_ = quadgk(μ -> pdf(Normal(μ, 1), -z) * cdf(Normal(μ, 1), -1.96) * pdf(prior, μ), -10, 10)

    @test isapprox(numerator(t)(prior), term_plus + term_minus; atol=1e-6)

    #test for Normal prior (Normal sample)
    for (μ0, τ) in ((0.0, 0.5), (0.0, 1.0), (0.4, 0.7), (-0.8, 1.2), (1.0, 2.0))
        prior = Normal(μ0, τ)
        for z in (-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0)
            t = Empirikos.ReplicationProbability(NormalSample(z, 1.0))
            val = t(prior)
            val_impl = numerator(t)(prior) / denominator(t)(prior)
            @test isapprox(val, val_impl; rtol=1e-11, atol=1e-12)

            val_exact = repl_closed_form_normalsample(z, μ0, τ)
            @test isapprox(val, val_exact; rtol=1e-10, atol=1e-12)
        end
    end

    #test for uniform prior (Normal sample)
    prior = Uniform(-10.0, 10.0)
    for z in (4.0, -4.0)
        t = Empirikos.ReplicationProbability(NormalSample(z, 1.0))

        if z >= 0
            num_quad, _ = quadgk(μ -> pdf(Normal(μ, 1), z) * ccdf(Normal(μ, 1),  1.96) * pdf(prior, μ), -10, 10)
        else
            num_quad, _ = quadgk(μ -> pdf(Normal(μ, 1), z) * cdf(Normal(μ, 1), -1.96) * pdf(prior, μ), -10, 10)
        end

        @test isapprox(numerator(t)(prior), num_quad; atol=1e-6)
    end
end

#Test for FutureCoverageProbability
function futcov_closed_form(z, μ, τ)
    σ = sqrt(1 + τ^2 / (1 + τ^2))
    mplus  = (τ^2 * z + μ) / (1 + τ^2)
    mminus = (-τ^2 * z + μ) / (1 + τ^2)
    σm = sqrt(1 + τ^2)
    fz   = pdf(Normal(μ, σm),  z)
    fneg = pdf(Normal(μ, σm), -z)

    prob_plus  = cdf(Normal(mplus,  σ),  z + 1.96) - cdf(Normal(mplus,  σ),  z - 1.96)
    prob_minus =  cdf(Normal(mminus, σ), -z + 1.96) - cdf(Normal(mminus, σ), -z - 1.96)

    num = fz * prob_plus + fneg * prob_minus
    den = fz + fneg
    return num / den
end

function futcov_closed_form_normalsample(z, μ, τ)
    σ = sqrt(1 + τ^2 / (1 + τ^2))
    m = (τ^2 * z + μ) / (1 + τ^2)

    return cdf(Normal(m, σ), z + 1.96) - cdf(Normal(m, σ), z - 1.96)
end


@testset "FutureCoverageProbability" begin
    #test for normal prior (folded normal)
    for (μ0, τ) in ((0.0, 0.5), (0.0, 1.0), (0.4, 0.7), (-0.8, 1.2), (1.0, 2.0))
        prior = Normal(μ0, τ)
        for z in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0)
            t = Empirikos.FutureCoverageProbability(FoldedNormalSample(abs(z)))
            val = t(prior)
            val_impl = numerator(t)(prior) / denominator(t)(prior)
            @test isapprox(val, val_impl; rtol=1e-11, atol=1e-12)

            val_exact = futcov_closed_form(abs(z), μ0, τ)
            @test isapprox(val, val_exact; rtol=1e-10, atol=1e-12)
        end
    end
    
    #test for uniform prior (folded normal)
    prior = Uniform(-10.0, 10.0)
    z = 4.0
    t = Empirikos.FutureCoverageProbability(FoldedNormalSample(z))

    term_plus, _ = quadgk(μ -> pdf(Normal(μ, 1),  z) *
        (cdf(Normal(μ, 1),  z + 1.96) - cdf(Normal(μ, 1),  z - 1.96)) * pdf(prior, μ), -10, 10)
    term_minus,_ = quadgk(μ -> pdf(Normal(μ, 1), -z) * 
        (cdf(Normal(μ, 1), -z + 1.96) - cdf(Normal(μ, 1), -z - 1.96)) * pdf(prior, μ), -10, 10)

    @test isapprox(numerator(t)(prior), term_plus + term_minus; atol=1e-6)

    #test for Normal prior (Normal sample)
    for (μ0, τ) in ((0.0, 0.5), (0.0, 1.0), (0.4, 0.7), (-0.8, 1.2), (1.0, 2.0))
        prior = Normal(μ0, τ)
        for z in (-3.0, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0)
            t = Empirikos.FutureCoverageProbability(NormalSample(z, 1.0))
            val = t(prior)
            val_impl = numerator(t)(prior) / denominator(t)(prior)
            @test isapprox(val, val_impl; rtol=1e-11, atol=1e-12)

            val_exact = futcov_closed_form_normalsample(z, μ0, τ)
            @test isapprox(val, val_exact; rtol=1e-10, atol=1e-12)
        end
    end

    #test for uniform prior (Normal sample)
    prior = Uniform(-10.0, 10.0)
    for z in (4.0, -4.0)
        t = Empirikos.FutureCoverageProbability(NormalSample(z, 1.0))
        num_quad, _ = quadgk(μ -> pdf(Normal(μ, 1), z) * 
        (cdf(Normal(μ, 1), z + 1.96) - cdf(Normal(μ, 1), z - 1.96))  * pdf(prior, μ), -10, 10)
        @test isapprox(numerator(t)(prior), num_quad; atol=1e-6)
    end
end

