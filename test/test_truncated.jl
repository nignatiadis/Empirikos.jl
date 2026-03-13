using Empirikos
using Test

@testset "_truncated" begin
    d = Normal()
    @testset "finite interval" begin
        td = Empirikos._truncated(d, Interval(0.0, 1.0))
        Z = cdf(d, 1.0) - cdf(d, 0.0)
        @test minimum(td) == 0.0
        @test maximum(td) == 1.0
        @test pdf(td, 0.5) ≈ pdf(d, 0.5) / Z
        @test cdf(td, 0.5) ≈ (cdf(d, 0.5) - cdf(d, 0.0)) / Z
    end
    @testset "right-bounded interval (-Inf, b]" begin
        td = Empirikos._truncated(d, Interval(-Inf, 1.0))
        @test minimum(td) == -Inf
        @test maximum(td) == 1.0
        @test pdf(td, 0.0) ≈ pdf(d, 0.0) / cdf(d, 1.0)
        @test cdf(td, 0.0) ≈ (cdf(d, 0.0) - cdf(d, -Inf)) / cdf(d, 1.0)
    end
    @testset "left-bounded interval [a, Inf)" begin
        td = Empirikos._truncated(d, Interval(-1.0, Inf))

        @test minimum(td) == -1.0
        @test maximum(td) == Inf
        @test pdf(td, 0.0) ≈ pdf(d, 0.0) / (1.0 - cdf(d, -1.0))
        @test cdf(td, 0.0) ≈ (cdf(d, 0.0) - cdf(d, -1.0)) / (1.0 - cdf(d, -1.0))
    end
    @testset "fully unbounded interval (-Inf, Inf)" begin
        td = Empirikos._truncated(d, Interval(-Inf, Inf))

        @test td == d
        @test pdf(td, 0.3) ≈ pdf(d, 0.3)
        @test cdf(td, 0.3) ≈ cdf(d, 0.3)
    end
    @testset "invalid left endpoint Inf" begin
        I = Interval(Inf, Inf)
        @test_throws ArgumentError Empirikos._truncated(d, I)
    end
    @testset "invalid right endpoint -Inf" begin
        I = Interval(-Inf, -Inf)
        @test_throws ArgumentError Empirikos._truncated(d, I)
    end
    @testset "reversed finite endpoints" begin
        I = Interval(2.0, 1.0)
        @test_throws Exception Empirikos._truncated(d, I)
    end
end

@testset "untilt" begin
    @testset "Mixture prior class" begin
        G = GaussianScaleMixtureClass([0.5, 1.0, 2.0])
        trunc_set = Interval(1.96, Inf)
        Z_trunc_set = FoldedNormalSample(trunc_set, 1.0)
        tilted = Empirikos.tilt(G, Z_trunc_set)
        untilted = Empirikos.untilt(tilted)
        @test components(untilted) == components(G)
    end
    @testset "MixtureModel of selection tilted: two components" begin
        d1 = Normal(0.0, 1.0)
        d2 = Normal(0.0, 2.0)
        G = GaussianScaleMixtureClass([1.0,2.0])
        trunc_set = Interval(1.96, Inf)
        Z_trunc_set = FoldedNormalSample(trunc_set, 1.0)
        tilted = Empirikos.tilt(G, Z_trunc_set)
        tilted_weights = [0.6, 0.4]
        tilted = tilted(tilted_weights)
        untilted = Empirikos.untilt(tilted)
        tilted_comps = components(tilted)
        untilted_comps = components(untilted)
        untilted_weights = probs(untilted)

        @test untilted_comps[1] == Empirikos.untilt(tilted_comps[1])
        @test untilted_comps[2] == Empirikos.untilt(tilted_comps[2])

        selprob = getfield.(tilted_comps, :selection_probability)
        expected_weights = tilted_weights ./ selprob
        expected_weights ./= sum(expected_weights)

        @test untilted_weights ≈ expected_weights
        @test sum(untilted_weights) ≈ 1.0

        @test untilted_comps[1] == Normal(0.0, 1.0)
        @test untilted_comps[2] == Normal(0.0, 2.0)
    end
    @testset "MixtureModel of selection tilted: three components" begin
        G = GaussianScaleMixtureClass([0.5, 1.0, 2.0])
        trunc_set = Interval(1.96, Inf)
        Z_trunc_set = FoldedNormalSample(trunc_set, 1.0)

        tilted_weights = [0.2, 0.3, 0.5]
        tilted = Empirikos.tilt(G, Z_trunc_set)(tilted_weights)
        untilted = Empirikos.untilt(tilted)

        tilted_comps = components(tilted)
        s = getfield.(tilted_comps, :selection_probability)

        expected_weights = tilted_weights ./ s
        expected_weights ./= sum(expected_weights)

        @test probs(untilted) ≈ expected_weights
        @test sum(probs(untilted)) ≈ 1.0

        comps = components(untilted)
        @test comps[1] == Normal(0.0, 0.5)
        @test comps[2] == Normal(0.0, 1.0)
        @test comps[3] == Normal(0.0, 2.0)
    end
end
