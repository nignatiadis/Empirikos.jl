module Empirikos

using Reexport

import Base:eltype, zero
using DataStructures
@reexport using Distributions
import Distributions:ntrials, pdf, support, location, cf, cdf

@reexport using Intervals
using JuMP
using LinearAlgebra
using Optim
using Statistics
import Statistics:std,var
@reexport using StatsBase
import StatsBase:loglikelihood, response, fit, nobs

using UnPack


include("ebayes_samples.jl")
include("ebayes_methods.jl")
include("ebayes_targets.jl")
include("mixtures.jl")
include("convex_priors.jl")
include("neighborhoods.jl")
include("samples/binomial.jl")
include("samples/normal.jl")


export EBayesSample,
       summarize,
       NormalSample,
       StandardNormalSample,
       BinomialSample,
       marginalize,
       summarize,
       likelihood,
       likelihood_distribution,
       EBInterval,
       nuisance_parameter,
       skedasticity,
       MethodOfMoments,
       ParametricMLE,
       PosteriorMean,
       PosteriorVariance,
       MarginalDensity,
       DiscretePriorClass,
       DvoretzkyKieferWolfowitz,
       linear_functional

end
