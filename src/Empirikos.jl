module Empirikos


import Base: broadcast, broadcast!, broadcasted, eltype, zero, <=
using DataStructures
using Distributions
import Distributions:
    ntrials, pdf, support, location, cf, cdf, ccdf, logpdf, logdiffcdf, logcdf, logccdf, components


import JuMP
import JuMP: @constraint, @variable, set_lower_bound, @expression,
    Model, @objective, optimize!, objective_value, set_objective,
    set_normalized_rhs, RotatedSecondOrderCone, SecondOrderCone, set_value

using KernelDensity
using LinearAlgebra
using LinearFractional
using LogExpFunctions
using MathOptInterface
using MultipleTesting
using ParameterJuMP
using QuadGK
using Random
using RangeHelpers
using RecipesBase
using Setfield
import SpecialFunctions: trigamma, digamma, polygamma, erfinv
using Statistics
import Statistics: std, var

using StatsBase
import StatsBase: loglikelihood, response, fit, nobs, weights, confint

using StatsDiscretizations
using UnPack

include("utils.jl")
include("ebayes_samples.jl")
include("compound.jl")
include("ebayes_methods.jl")
include("ebayes_targets.jl")
include("mixtures.jl")
include("convex_priors.jl")
include("flocalizations.jl")
include("NPMLE.jl")
include("samples/binomial.jl")
include("samples/bivariate_binomial.jl")
include("samples/normal.jl")
include("samples/poisson.jl")
include("samples/truncatedpoisson.jl")
include("samples/noncentralhypergeometric.jl")
include("samples/scaledchisquare.jl")
include("samples/normalchisquare.jl")
include("samples/foldednormal.jl")
include("samples/truncated.jl")
include("samples/replicated_sample.jl")

include("autoconvexclass.jl")


include("example_priors.jl")
include("confidence_interval_tools.jl")
include("flocalization_intervals.jl")
include("flocalization_kde.jl")
include("amari.jl")
include("empirical_partially_bayes_multiple_testing.jl")


include("datasets/LordCressie/LordCressie.jl")
include("datasets/Prostate/Prostate.jl")
include("datasets/Neighborhoods/neighborhoods.jl")
include("datasets/Butterfly/Butterfly.jl")
include("datasets/BertrandMullainathan/BertrandMullainathan.jl")
include("datasets/ArceoGomezCamposVasquez/ArceoGomezCamposVasquez.jl")

include("datasets/Surgery/Surgery.jl")
include("datasets/CollinsLangman/CollinsLangman.jl")
include("datasets/CressieSeheult/CressieSeheult.jl")
include("datasets/EfronMorrisBaseball/EfronMorrisBaseball.jl")
include("datasets/Bichsel/Bichsel.jl")
include("datasets/Thyrion/Thyrion.jl")
include("datasets/Tacks/Tacks.jl")
include("datasets/Shakespeare/Shakespeare.jl")
include("datasets/PsychologyReproducibility/PsychologyReproducibility.jl")



export EBayesSample,
    NormalSample,
    StandardNormalSample,
    FoldedNormalSample,
    BinomialSample,
    BivariateBinomialSample,
    PoissonSample,
    TruncatedPoissonSample,
    ScaledChiSquareSample,
    NormalChiSquareSample,
    NonCentralHypergeometricSample,
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
    nominal_alpha

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



export FLocalizationInterval,
       AMARI

end
