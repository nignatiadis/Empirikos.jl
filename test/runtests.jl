using Empirikos
using Test
using InteractiveUtils

@testset "EBayes Intervals" begin
    include("test_intervals.jl")
end

@testset "Binomial samples" begin
    include("test_binomial.jl")
end
