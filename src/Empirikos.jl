module Empirikos

using Reexport

import Base:eltype, zero
@reexport using Distributions
import Distributions:ntrials, pdf, support

@reexport using Intervals

using Statistics
import Statistics:std,var
@reexport using StatsBase
import StatsBase:loglikelihood, response, fit

using UnPack


include("ebayes_samples.jl")
include("ebayes_methods.jl")
include("samples/binomial.jl")
include("samples/normal.jl")

export EBayesSample,
       NormalSample,
       StandardNormalSample,
       BinomialSample,
       marginalize,
       likelihood,
       likelihood_distribution,
       EBInterval,
       nuisance_parameter,
       skedasticity,
       MethodOfMoments

end
