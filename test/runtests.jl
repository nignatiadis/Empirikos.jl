using Empirikos
using Test
using InteractiveUtils
using FiniteDifferences
using ForwardDiff
using Hypatia
using QuadGK
using Documenter
using CSV
using Distributions
using Random

# Doctests

#DocMeta.setdocmeta!(Empirikos, :DocTestSetup, :(using Empirikos); recursive = true)
#doctest(Empirikos)

# Other tests



@testset "Test ci tools" begin
    include("test_ci_tools.jl")
end

#@testset "Dict function" begin
#    include("test_dict_function.jl")
#end

@testset "EBayes Intervals" begin
    include("test_intervals.jl")
end


@testset "Posterior" begin
    include("test_posterior.jl")
end


@testset "Binomial samples" begin
    include("test_binomial.jl")
end

@testset "Normal samples" begin
    include("test_normal.jl")
end

@testset "FoldedNormal samples" begin
    include("test_foldednormal.jl")
end

@testset "Truncated Poisson samples" begin
    include("test_truncated_poisson.jl")
end

@testset "Normal-Chisquare samples" begin
    include("test_t_normal_chisquare.jl")
end

@testset "Test targets" begin
    include("test_targets.jl")
end

@testset "Chi-squared F localization" begin
    include("test_lord_cressie.jl")
end

@testset "Further F localization interval tests" begin
    include("test_flocalization_intervals.jl")
end

@testset "Kernel density tests" begin
    include("test_kernel_density.jl")
end

@testset "Bernoulli samples" begin
    include("test_bernoulli.jl")
end

@testset "Two kinds of modulus problems" begin
    include("test_modulus.jl")
end

@testset "Compound and DKW F localization" begin
    include("test_compound.jl")
end

@testset "Noncentral hypergeometric" begin
    include("test_noncentralhypergeometric.jl")
end

@testset "Empirical partially Bayes" begin
    include("test_empirical_partially_bayes.jl")
end
