using Empirikos
using Test
using Distributions
@testset "Folded Distribution" begin
    # Symmetric normal
    d = Normal(0.0, 1.0)
    folded_d = Empirikos.fold(d)
    # Asymmetric normal
    asy = Normal(1, 2)
    folded_asy= Empirikos.fold(asy)
    # Mixture of Asymmetric Normals
    comp = [Normal(-2.0, 0.5), Normal(1.5, 1.2)]
    w    = [0.3, 0.7]
    mix  = MixtureModel(comp, w)    
    fmx  = Empirikos.fold(mix)
    @testset "Basic Properties" begin
        @test Empirikos.unfold(folded_d) == d
        @test Empirikos.minimum(folded_d) == 0.0
        @test Empirikos.maximum(folded_d) == Inf 
        
        @test Empirikos.unfold(fmx) == mix
        @test Empirikos.minimum(fmx) == 0.0
        @test Empirikos.maximum(fmx) == Inf
    end

    @testset "PDF" begin
       
        @test Empirikos.pdf(folded_d, 1.0) == 2 * pdf(d, 1.0)

        @test Empirikos.pdf(folded_d, 0.0) == 2 * pdf(d, 0.0)
   
        @test Empirikos.pdf(folded_d, -1.0) == 0.0
        
        # Asymmetric normal
        @test Empirikos.pdf(folded_asy, 1.0) == pdf(asy, 1.0) + pdf(asy, -1.0)

        @test Empirikos.pdf(folded_asy, 0.0) == 2 * pdf(asy, 0.0)
   
        @test Empirikos.pdf(folded_asy, -1.0) == 0.0

        # Mixture of Asymmetric Normals
        ys_pos = [0.0, 0.2, 0.8, 1.7, 3.5, 6.0]
        @test Empirikos.pdf(fmx, -1.0) == 0.0
        @test Empirikos.pdf(fmx,  0.0) ≈ 2*pdf(mix, 0.0) atol=1e-12
        @test all(y -> isapprox(Empirikos.pdf(fmx, y), 0.3* (pdf(comp[1], y) + pdf(comp[1], -y))+ 0.7*(pdf(comp[2], y) + pdf(comp[2], -y));
                             rtol=1e-10, atol=1e-12), ys_pos)
    end


    @testset "CDF" begin
        @test Empirikos.cdf(folded_d, 1.0) == cdf(d, 1.0) - cdf(d, -1.0)
        @test Empirikos.cdf(folded_d, 0.0) == 0.0
        
        # Asymmetric normal
        @test Empirikos.cdf(folded_asy, -1.0) == 0.0
        @test Empirikos.cdf(folded_asy, 2.0) == cdf(asy, 2.0) - cdf(asy, -2.0)

        # Mixture of Asymmetric Normals
        ys_pos = [0.0, 0.2, 0.8, 1.7, 3.5, 6.0]
        @test Empirikos.cdf(fmx, -0.1) == 0.0
        @test Empirikos.cdf(fmx,  0.0) == 0.0
        @test all(y -> isapprox(Empirikos.cdf(fmx, y), 0.3*(cdf(comp[1], y) - cdf(comp[1], -y))+ 0.7*(cdf(comp[2], y) - cdf(comp[2], -y));
                             rtol=1e-10, atol=1e-12), ys_pos)
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

        @test isapprox(Empirikos.logpdf(folded_asy, 3.0), (log(Empirikos.pdf(asy, 3.0) + Empirikos.pdf(asy, -3.0)));atol=1e-8, rtol=1e-8)
    end

    @testset "Uniform" begin
  
        u = Uniform(-1, 2)
        folded_u = Empirikos.fold(u)
        @test Empirikos.minimum(folded_u) == 0.0
        @test Empirikos.maximum(folded_u) == 2.0  
        @test Empirikos.pdf(folded_u, 0.5) == 2 * pdf(u, 0.5)  
    end 

    @testset "Folded Normal Computations" begin
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

        @testset "marginalize(FoldedNormalSample, Mixture{Normal})" begin
            z, σ = 1.4, 0.8
            Zf = FoldedNormalSample(z, σ)

            comps = [Normal(-0.8, 0.7), Normal(1.5, 0.9), Normal(0.2, 0.4)]
            w     = [0.2, 0.5, 0.3]
            prior = MixtureModel(comps, w)
            got = marginalize(Zf, prior)


            pred_comps = [Normal(mean(c), sqrt(σ^2 + var(c))) for c in comps]
            pred = MixtureModel(pred_comps, w)


            for i in 1:3
                @test isapprox(mean(Empirikos.unfold(got.components[i])), mean(pred.components[i]); atol=1e-11, rtol=1e-11)
                @test isapprox(std(Empirikos.unfold(got.components[i])),  std(pred.components[i]);  atol=1e-11, rtol=1e-11)
            end
        
            ys = [0.0, 0.2, 0.9, 1.8, 3.4, 6.0]

            @test all(y -> isapprox(Empirikos.pdf(got, y), pdf(pred, y) + pdf(pred, -y);
                             rtol=1e-11, atol=1e-12), ys)
            @test all(y -> isapprox(Empirikos.cdf(got, y), cdf(pred, y) - cdf(pred, -y);
                             rtol=1e-11, atol=1e-12), ys)

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


        @testset "posterior(FoldedNormalSample, Mixture{Normal})" begin
            z, σ = 1.7, 0.9
            Zf = FoldedNormalSample(z, σ)
            comps = [Normal(-0.6, 0.5), Normal(1.2, 0.7), Normal(0.3, 1.1)]
            w     = [0.25, 0.55, 0.20]
            prior = MixtureModel(comps, w)
            post = Empirikos.posterior(Zf, prior)
            μ = [mean(c) for c in comps]
            τ = [std(c)  for c in comps]
            s2 = @. 1 / (1/σ^2 + 1/τ^2)
            s  = sqrt.(s2)
            mpos = @. s2 * ( +z/σ^2 + μ/τ^2 )
            mneg = @. s2 * ( -z/σ^2 + μ/τ^2 )
            spred = @. sqrt(σ^2 + τ^2)
            K = length(comps)
            out_w_unnorm = [ w[k] * ( pdf(Normal(μ[k], spred[k]), +z) +
                              pdf(Normal(μ[k], spred[k]), -z) )
                     for k in 1:K ]
            Znorm = sum(out_w_unnorm)
            out_w_exp = out_w_unnorm ./ Znorm

            in_w_exp = [ let wpos = pdf(Normal(μ[k], spred[k]), +z),
                     wneg = pdf(Normal(μ[k], spred[k]), -z)
                  [wpos/(wpos+wneg), wneg/(wpos+wneg)]
                 end for k in 1:K ]

            @test isapprox(probs(post), out_w_exp; rtol=1e-12, atol=1e-12)

            for k in 1:K
                mk = post.components[k]
                inner = [(mean(mk.components[j]), std(mk.components[j]), probs(mk)[j]) for j in 1:2]

                exp_pairs = [(mpos[k], s[k], in_w_exp[k][1]),
                     (mneg[k], s[k], in_w_exp[k][2])]

                sortbymean(v) = sort(v; by = x -> x[1])
                inner_s = sortbymean(inner)
                exp_s   = sortbymean(exp_pairs)
                for j in 1:2
                    μhat, σhat, ŵ = inner_s[j]
                    μexp, σexp, wexp = exp_s[j]
                    @test isapprox(μhat, μexp; rtol=1e-11, atol=1e-11)
                    @test isapprox(σhat, σexp; rtol=1e-11, atol=1e-11)
                    @test isapprox(ŵ,   wexp; rtol=1e-12, atol=1e-12)
                end
            end
    end
        @testset "marginalize(FoldedNormalSample, Uniform)" begin
            # Symmetric uniform prior
            z  = 1.1
            σ  = 0.9
            Zf = FoldedNormalSample(z, σ)
            a = 2.5
            uni_sym = Uniform(-a, a)
            got = Empirikos.marginalize(Zf, uni_sym)
            un = Empirikos.UniformNormal(-2.5, 2.5, σ)
            pdf_expected(y) = y < 0 ? 0.0 : pdf(un, y) + pdf(un, y)
            cdf_expected(y) = y ≤ 0 ? 0.0 :
            (cdf(un, y) - cdf(un, 0.0)) + (cdf(un, y) - cdf(un, 0.0))
            ys = [0.0, 0.2, 0.7, 1.5, 3.0, 6.0]
            @test pdf(got, -0.3) == 0.0
            @test cdf(got, -0.3) == 0.0
            @test all(y -> isapprox(pdf(got, y),  pdf_expected(y);  rtol=1e-8, atol=1e-10), ys)
            @test all(y -> isapprox(cdf(got, y),  cdf_expected(y);  rtol=1e-8, atol=1e-10), ys)
            
            # Asymmetric uniform prior
            uni_asym = Uniform(-2.0, 3.0)
            got_asym = Empirikos.marginalize(Zf, uni_asym)
            un_L = Empirikos.UniformNormal(-2.0, 3.0, σ)
            un_R = Empirikos.UniformNormal(-3.0, 2.0, σ)  # reflected interval
            pdf_expected(y) = y < 0 ? 0.0 : pdf(un_L, y) + pdf(un_R, y)
            cdf_expected(y) = y ≤ 0 ? 0.0 :
            (cdf(un_L, y) - cdf(un_L, 0.0)) + (cdf(un_R, y) - cdf(un_R, 0.0))
            ys2 = [0.0, 0.3, 1.0, 2.2, 4.5, 7.0]
            @test pdf(got_asym, -0.3) == 0.0
            @test cdf(got_asym, -0.3) == 0.0
            @test all(y -> isapprox(pdf(got_asym, y),  pdf_expected(y);  rtol=1e-8, atol=1e-10), ys2)
            @test all(y -> isapprox(cdf(got_asym, y),  cdf_expected(y);  rtol=1e-8, atol=1e-10), ys2)
        end
    end
end



