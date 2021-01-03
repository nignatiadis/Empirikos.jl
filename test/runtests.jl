using Empirikos
using Test
using InteractiveUtils
using FiniteDifferences
using ForwardDiff
using Hypatia
using QuadGK

@testset "Test set default" begin
    include("test_set_defaults.jl")
end

@testset "Dict function" begin
    include("test_dict_function.jl")
end

@testset "EBayes Intervals" begin
    include("test_intervals.jl")
end

@testset "Binomial samples" begin
    include("test_binomial.jl")
end

@testset "Normal samples" begin
    include("test_normal.jl")
end

@testset "Test targets" begin
    include("test_targets.jl")
end

@testset "Chi-squared F localization" begin
    include("test_lord_cressie.jl")
end
