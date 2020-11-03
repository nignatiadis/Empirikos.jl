module Empirikos

using Reexport

import Base: broadcast, broadcast!, broadcasted, eltype, zero, <=
using DataStructures
@reexport using Distributions
import Distributions:
    ntrials, pdf, support, location, cf, cdf, ccdf, logpdf, logdiffcdf, logccdf, components
using Expectations

@reexport using Intervals
using JuMP
using LinearAlgebra
using MathOptInterface
using Optim
using Setfield
using Statistics
import Statistics: std, var
@reexport using StatsBase
import StatsBase: loglikelihood, response, fit, nobs

using UnPack

include("dict_function.jl")
include("ebayes_samples.jl")
include("interval_discretizer.jl")
include("ebayes_methods.jl")
include("ebayes_targets.jl")
include("mixtures.jl")
include("convex_priors.jl")
include("neighborhoods.jl")
include("NPMLE.jl")
include("samples/binomial.jl")
include("samples/normal.jl")
include("samples/poisson.jl")


export EBayesSample,
    summarize,
    NormalSample,
    StandardNormalSample,
    BinomialSample,
    PoissonSample,
    marginalize,
    discretize,
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
    MixturePriorClass,
    NPMLE

# neighborhoods
export DvoretzkyKieferWolfowitz

# utilities
export DictFunction

end
