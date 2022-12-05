module Empirikos

using Reexport

import Base: broadcast, broadcast!, broadcasted, eltype, zero, <=
using DataStructures
@reexport using Distributions
import Distributions:
    ntrials, pdf, support, location, cf, cdf, ccdf, logpdf, logdiffcdf, logccdf, components

import Intervals: Interval, Closed, Open, Unbounded, Bounded, AbstractInterval, isbounded,
    RightEndpoint
export Interval, Closed, Open, Unbounded # instead of @reexport

import JuMP
import JuMP: @constraint, @variable, set_lower_bound, @expression,
    Model, @objective, optimize!, objective_value, set_objective,
    set_normalized_rhs, RotatedSecondOrderCone, SecondOrderCone, set_value

using KernelDensity
using LinearAlgebra
using LinearFractional
using MathOptInterface
using ParameterJuMP
using QuadGK
using Random
using RangeHelpers
using RecipesBase
using Setfield
import SpecialFunctions: trigamma, digamma, polygamma
using Statistics
import Statistics: std, var

using StatsBase
import StatsBase: loglikelihood, response, fit, nobs, weights, confint

using UnPack

include("set_defaults.jl")
include("ebayes_samples.jl")
include("compound.jl")
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
include("samples/truncatedpoisson.jl")
include("samples/noncentralhypergeometric.jl")
include("samples/scaledchisquare.jl")

include("samples/foldednormal.jl")
include("example_priors.jl")
include("confidence_interval_tools.jl")
include("flocalization_intervals.jl")
include("flocalization_kde.jl")
include("amari.jl")


include("datasets/LordCressie/LordCressie.jl")
include("datasets/Prostate/Prostate.jl")
include("datasets/Neighborhoods/neighborhoods.jl")
include("datasets/Butterfly/Butterfly.jl")
include("datasets/Surgery/Surgery.jl")
include("datasets/CollinsLangman/CollinsLangman.jl")
include("datasets/CressieSeheult/CressieSeheult.jl")
include("datasets/Bichsel/Bichsel.jl")
include("datasets/Tacks/tacks.jl")



export EBayesSample,
    NormalSample,
    StandardNormalSample,
    BinomialSample,
    PoissonSample,
    TruncatedPoissonSample,
    ScaledChiSquareSample,
    marginalize,
    compound,
    discretize,
    summarize,
    likelihood,
    likelihood_distribution,
    EBInterval,
    nuisance_parameter,
    skedasticity,
    MethodOfMoments,
    PosteriorMean,
    PosteriorVariance,
    MarginalDensity,
    PriorDensity,
    DiscretePriorClass,
    MixturePriorClass,
    GaussianScaleMixtureClass,
    NPMLE,
    nominal_alpha,
    integer_discretizer,
    interval_discretizer

export loglikelihood,
    response,
    fit,
    nobs,
    weights,
    confint

# F-Localizations
export DvoretzkyKieferWolfowitz,
    ChiSquaredFLocalization,
    InfinityNormDensityBand

# utilities
export DictFunction

# default

export DataBasedDefault

export FLocalizationInterval,
       AMARI

end
