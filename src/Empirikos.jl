module Empirikos

using Reexport

import Base:eltype, zero
@reexport using Distributions
import Distributions:ntrials, pdf, support, location

@reexport using Intervals

using Optim
using Statistics
import Statistics:std,var
@reexport using StatsBase
import StatsBase:loglikelihood, response, fit

using UnPack


include("ebayes_samples.jl")
include("ebayes_methods.jl")
include("ebayes_targets.jl")
include("samples/binomial.jl")
include("samples/normal.jl")


export EBayesSample,
       summarize,
       NormalSample,
       StandardNormalSample,
       BinomialSample,
       marginalize,
       likelihood,
       likelihood_distribution,
       EBInterval,
       nuisance_parameter,
       skedasticity,
       MethodOfMoments,
       ParametricMLE,
       PosteriorMean,
       PosteriorVariance

end
