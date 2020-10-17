using Empirikos
using Test
using InteractiveUtils

@testset "Dict function" begin
    include("test_dict_function.jl")
end

@testset "EBayes Intervals" begin
    include("test_intervals.jl")
end

@testset "Binomial samples" begin
    include("test_binomial.jl")
end
