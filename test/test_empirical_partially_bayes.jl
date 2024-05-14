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


# Tests on Mellin transform

G1 = Empirikos.InverseScaledChiSquare(3.1, 2.0)
G2 = Empirikos.InverseScaledChiSquare(4.0, 2.5)

function mellin(G::Empirikos.InverseScaledChiSquare, c, t)
    ν0 = G.ν
    s² = G.σ²
    exp(log(2/ν0/s²)*(- c + 1 - im*t)) * gamma(ν0/2 - c + 1 - im*t) / gamma(ν0/2)
end

function mellin_quadgk(G::Distribution, c, t)
    quadgk(u->pdf(G,u)*exp(log(u)*(c-1 + im*t)), 0, Inf)[1]
end 

@test mellin(G1, 1/2, 0.0) ≈ mellin_quadgk(G1, 1/2, 0.0)
@test mellin(G2, 1/2, 0.0) ≈ mellin_quadgk(G2, 1/2, 0.0)
@test mellin(G2, 1/2, 3.0) ≈ mellin_quadgk(G2, 1/2, 3.0)
@test mellin(G2, 1/2, -3.0) ≈ mellin_quadgk(G2, 1/2, -3.0)
@test mellin(G2, 1, 1.0) ≈ mellin_quadgk(G2, 1, 1.0)


# for the likelihood function under σ=1
function mellin(S::ScaledChiSquareSample, c, t)
    ν = S.ν
    exp(log(2/ν)*(c - 1 + im*t)) * gamma(c + ν/2 - 1 + im*t) / gamma(ν/2)
end

function mellin_quadgk(S::ScaledChiSquareSample, c, t)
    d = Empirikos.likelihood_distribution(S, 1.0)
    quadgk(u->pdf(d,u)*exp(log(u)*(c-1 + im*t)), 0, Inf)[1]
end

S2 = ScaledChiSquareSample(1.0, 2)
S3 = ScaledChiSquareSample(1.0, 3)

function ratio_fun(ν, c, t)
    (ν/(ν+1))^(c-1) * gamma(ν/2) / gamma((ν+1)/2) * sqrt(t)
end
ts = 0.0:0.1:10.0

mell_S2_ts = mellin.(S2, 1/2, ts)
mell_S3_ts = mellin.(S3, 1/2, ts)

#plot(ts, abs.(mell_S3_ts ./ mell_S2_ts))
#plot!(ts, ratio_fun.(2, 1/2, ts))

@test mellin(S2, 1/2, 2.0) ≈ mellin_quadgk(S2, 1/2, 2.0)
@test  mellin(S2, 1.1, 2.0) ≈ mellin_quadgk(S2, 1.1, 2.0)
@test mellin(S3, 1/2, 2.0) ≈ mellin_quadgk(S3, 1/2, 2.0)
@test  mellin(S3, 1.1, 2.0) ≈ mellin_quadgk(S3, 1.1, 2.0)

F_1_2_marg = Empirikos.marginalize(S2, G1)
@test mellin_quadgk(F_1_2_marg, 1/2, 2.0) ≈  mellin(G1, 1/2, 2.0) * mellin(S2, 1/2, 2.0) 

F_2_2_marg = Empirikos.marginalize(S2, G2)
@test mellin_quadgk(F_2_2_marg, 1/2, 2.0) ≈  mellin(G2, 1/2, 2.0) * mellin(S2, 1/2, 2.0) 


F_1_3_marg = Empirikos.marginalize(S3, G1)
@test mellin_quadgk(F_1_3_marg, 1/2, 2.0) ≈  mellin(G1, 1/2, 2.0) * mellin(S3, 1/2, 2.0) 

F_2_3_marg = Empirikos.marginalize(S3, G2)
@test mellin_quadgk(F_2_3_marg, 1/2, 2.0) ≈  mellin(G2, 1/2, 2.0) * mellin(S3, 1/2, 2.0) 



@test quadgk(t ->  abs2((mellin(G1, 1/2, t) - mellin(G2, 1/2, t))* mellin(S2, 1/2, t)), -Inf, +Inf)[1]/2/π ≈ quadgk(t ->  abs2(pdf(F_1_2_marg, t ) - pdf(F_2_2_marg, t)), 0, +Inf)[1]

@test quadgk(t ->  abs2((mellin(G1, 1/2, t) - mellin(G2, 1/2, t))* mellin(S3, 1/2, t)), -Inf, +Inf)[1]/2/π ≈ quadgk(t ->  abs2(pdf(F_1_3_marg, t ) - pdf(F_2_3_marg, t)), 0, +Inf)[1]



G3(s) =  Empirikos.InverseScaledChiSquare(3.1, s)
s_grid = 2.01:0.001:2.2

function compute_l2_distance(G, H, dof)
    S = ScaledChiSquareSample(1.0, dof)
    FG = Empirikos.marginalize(S, G)
    FH = Empirikos.marginalize(S, H)
    quadgk(t ->  abs2(pdf(FG, t ) - pdf(FH, t)), 0, +Inf)[1]
end

function compute_l2_distance_mellin(G, H, dof)
    S = ScaledChiSquareSample(1.0, dof)
    quadgk(t ->  abs2((mellin(G, 1/2, t) - mellin(H, 1/2, t))* mellin(S, 1/2, t)), -Inf, +Inf)[1]/2/π 
end


dists_at_2 = compute_l2_distance.(G1, G3.(s_grid), 2)
dists_at_3 = compute_l2_distance.(G1, G3.(s_grid), 3)

dists_at_2_mellin = compute_l2_distance_mellin.(G1, G3.(s_grid), 2)
@test dists_at_2 ≈ dists_at_2_mellin

dists_at_3_mellin = compute_l2_distance_mellin.(G1, G3.(s_grid), 3)
@test dists_at_3 ≈ dists_at_3_mellin

dists_at_10 = compute_l2_distance.(G1, G3.(s_grid), 10)
dists_at_10_mellin = compute_l2_distance_mellin.(G1, G3.(s_grid), 10)
@test dists_at_10 ≈ dists_at_10_mellin



#plot(dists_at_2, dists_at_3, seriestype=:scatter)
#plot!([0.0;dists_at_2[end]], [0.0;dists_at_2[end]])

#plot(dists_at_3, dists_at_2 .* log.(1 ./ dists_at_2), seriestype=:scatter)
#plot!([0.0;0.0012], [0.0;0.0012])

all(dists_at_3 .< dists_at_2 .* log.(1 ./ dists_at_2))
# TODO: also add some more tests, also e.g. on Limma.


ν = 20
u=1e-7
@test cdf(Chisq(ν), u)*u^(-ν/2) ≈ 2^(-ν/2)/gamma(1 + (ν/2)) rtol = 1e-5