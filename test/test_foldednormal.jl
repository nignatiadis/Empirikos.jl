using Empirikos
using Test
using Distributions
using IntervalSets
using Statistics
@testset "Folded Distribution" begin
    d = Normal(0.0, 1.0)
    folded_d = Empirikos.fold(d)

    @testset "Basic Properties" begin
        @test Empirikos.unfold(folded_d) == d
        @test Empirikos.minimum(folded_d) == 0.0
        @test Empirikos.maximum(folded_d) == Inf 
    end

    @testset "PDF" begin
       
        @test Empirikos.pdf(folded_d, 1.0) == 2 * pdf(d, 1.0)

        @test Empirikos.pdf(folded_d, 0.0) == 2 * pdf(d, 0.0)
   
        @test Empirikos.pdf(folded_d, -1.0) == 0.0
        asy = Normal(1, 2)
        folded_asy= Empirikos.fold(asy)
        @test Empirikos.pdf(folded_asy, 1.0) == pdf(asy, 1.0) + pdf(asy, -1.0)

        @test Empirikos.pdf(folded_asy, 0.0) == 2 * pdf(asy, 0.0)
   
        @test Empirikos.pdf(folded_asy, -1.0) == 0.0
    end


    @testset "CDF" begin
        @test Empirikos.cdf(folded_d, 1.0) == cdf(d, 1.0) - cdf(d, -1.0)

        @test Empirikos.cdf(folded_d, 0.0) == 0.0
    end

    @testset "Folded Normal quantile" begin
       
        d0 = Empirikos.Folded(Normal(0.0, 1.0))
        x0 = Empirikos.quantile(d0, 0.5)
        @test isapprox(x0, sqrt(Empirikos.quantile(Chisq(1), 0.5)); atol=1e-8)
    
        for (μ, σ, q) in (
                (0.0, 1.0, 0.1),
                (0.0, 1.0, 0.9),
                (1.0, 2.0, 0.25),
                (1.5, 0.5, 0.75),
                (−2.0, 3.0, 0.5),
            )
            d = Empirikos.Folded(Normal(μ, σ))
            x = Empirikos.quantile(d, q)
            @test x ≥ 0
            @test isapprox(Empirikos.cdf(d, x), q; atol=1e-8, rtol=1e-8)
        end

        deg = Empirikos.Folded(Normal(5.0, 0.0))
        @test Empirikos.quantile(deg, 0.0) == abs(5.0)
        @test Empirikos.quantile(deg, 0.5) == abs(5.0)
        @test Empirikos.quantile(deg, 1.0) == abs(5.0)
    end

    @testset "Invalid Constructions" begin
        @test_throws DomainError Empirikos.FoldedNormalSample(-1.0, 0.5)  
        @test_throws DomainError Empirikos.FoldedNormalSample(-1.0) 
        @test_throws DomainError Empirikos.FoldedNormalSample(2.0, -0.5)
    end
    @testset "LogPDF" begin
        d = Normal(0.0, 1.0)
        folded_d = Empirikos.fold(d)
        @test Empirikos.logpdf(folded_d, 1.0) == (log(2) + log(Empirikos.pdf(d, -1.0)))
        @test Empirikos.logpdf(folded_d, -1.0) == -Inf
    end

    @testset "Uniform" begin
  
        u = Uniform(-1, 2)
        folded_u = Empirikos.fold(u)
        @test Empirikos.minimum(folded_u) == 0.0
        @test Empirikos.maximum(folded_u) == 2.0  
        @test Empirikos.pdf(folded_u, 0.5) == 2 * pdf(u, 0.5)  
    end 

    @testset "Folded Normal Computations" begin
        fz = Empirikos.FoldedNormalSample(2.0)
        normal_prior = Normal(0, 1)
        folded_prior = Empirikos.fold(normal_prior)
        sym_uniform = Uniform(-1, 1)
        asym_uniform = Uniform(0, 1)
    
        @testset "likelihood_distribution for FoldedNormalSample" begin
            z  = 2.3
            σ  = 1.2
            μ  = -0.7
            Zf = FoldedNormalSample(z, σ)

            dbn = likelihood_distribution(Zf, μ)

            @test isapprox(mean(Empirikos.unfold(dbn)), μ; atol=1e-10, rtol=1e-10)
            @test isapprox(std(Empirikos.unfold(dbn)),  σ; atol=1e-10, rtol=1e-10)
        end
        @testset "marginalize(FoldedNormalSample, Normal)" begin
            z  = 1.7
            σ  = 1.0
            μ0 = 0.5
            τ0 = 1.3
            Zf = FoldedNormalSample(z, σ)

            got = marginalize(Zf, Normal(μ0, τ0))

            pred = Normal(μ0, sqrt(σ^2 + τ0^2))
            @test isapprox(mean(Empirikos.unfold(got)), mean(pred); atol=1e-10, rtol=1e-10)
            @test isapprox(std(Empirikos.unfold(got)),  std(pred);  atol=1e-10, rtol=1e-10)
        end
        @testset "marginalize(FoldedNormalSample, Folded{Normal})" begin
            z  = 0.9
            σ  = 0.7
            μ0 = -0.2
            τ0 = 1.1
            Zf = FoldedNormalSample(z, σ)

            folded_prior = Empirikos.fold(Normal(μ0, τ0))
            got = marginalize(Zf, folded_prior)

            pred = Normal(μ0, sqrt(σ^2 + τ0^2))
            @test isapprox(mean(Empirikos.unfold(got)), mean(pred); atol=1e-10, rtol=1e-10)
            @test isapprox(std(Empirikos.unfold(got)),  std(pred);  atol=1e-10, rtol=1e-10)
        end
        @testset "posterior(FoldedNormalSample, Normal)" begin
            z  = 2.0
            σ  = 1.0
            μ0 = 1.0
            τ0 = 1.0
            Zf = FoldedNormalSample(z, σ)
            pr = Normal(μ0, τ0)

            post = Empirikos.posterior(Zf, pr)
            @test length(post.components) == 2
            @test length(post.prior.p) == 2
            @test isapprox(sum(post.prior.p), 1.0; atol=1e-12)

            s2 = 1 / (1/σ^2 + 1/τ0^2)
            s  = sqrt(s2)
            mpos = s2 * ( (+z)/σ^2 + μ0/τ0^2 )
            mneg = s2 * ( (-z)/σ^2 + μ0/τ0^2 )

            @test isapprox(mean(post.components[1]), mpos; atol=1e-10, rtol=1e-10)
            @test isapprox(std(post.components[1]),  s;    atol=1e-10, rtol=1e-10)
            @test isapprox(mean(post.components[2]), mneg; atol=1e-10, rtol=1e-10)
            @test isapprox(std(post.components[2]),  s;    atol=1e-10, rtol=1e-10)
            sp = sqrt(σ^2 + τ0^2)
            wpos = pdf(Normal(μ0, sp), +z)
            wneg = pdf(Normal(μ0, sp), -z)
            wsum = wpos + wneg
            @test isapprox(post.prior.p[1], wpos/wsum; atol=1e-12, rtol=1e-12)
            @test isapprox(post.prior.p[2], wneg/wsum; atol=1e-12, rtol=1e-12)
        end
        @testset "marginalize(FoldedNormalSample, Uniform)" begin
            z  = 1.1
            σ  = 0.9
            Zf = FoldedNormalSample(z, σ)
            a = 2.5
            uni_sym = Uniform(-a, a)
            got = Empirikos.marginalize(Zf, uni_sym)

            uni_asym = Uniform(-2.0, 3.0)
            @test_throws DomainError Empirikos.marginalize(Zf, uni_asym)
        end
    end
end