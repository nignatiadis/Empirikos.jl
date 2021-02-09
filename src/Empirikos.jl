module Empirikos

using Reexport

import Base: broadcast, broadcast!, broadcasted, eltype, zero, <=
using DataStructures
@reexport using Distributions
import Distributions:
    ntrials, pdf, support, location, cf, cdf, ccdf, logpdf, logdiffcdf, logccdf, components

@reexport using Intervals
using JuMP
using KernelDensity
using LinearAlgebra
using LinearFractional
using MathOptInterface
using Optim
using ParameterJuMP
using QuadGK
using Random
using RecipesBase
using Roots
using Setfield
using Statistics
import Statistics: std, var
@reexport using StatsBase
import StatsBase: loglikelihood, response, fit, nobs, weights

using UnPack

include("set_defaults.jl")
include("ebayes_samples.jl")
include("interval_discretizer.jl")
include("dict_function.jl")
include("ebayes_methods.jl")
include("ebayes_targets.jl")
include("mixtures.jl")
include("convex_priors.jl")
include("flocalizations.jl")
include("NPMLE.jl")
include("samples/binomial.jl")
include("samples/normal.jl")
include("samples/poisson.jl")
include("example_priors.jl")
include("confidence_interval_tools.jl")
include("flocalization_intervals.jl")
include("flocalization_kde.jl")
include("local_linear_minimax.jl")

include("datasets/LordCressie/LordCressie.jl")
include("datasets/Prostate/Prostate.jl")



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
    GaussianScaleMixtureClass,
    NPMLE,
    nominal_alpha

# F-Localizations
export DvoretzkyKieferWolfowitz,
    ChiSquaredFLocalization

# utilities
export DictFunction

# default

export DataBasedDefault

export FLocalizationInterval

end
