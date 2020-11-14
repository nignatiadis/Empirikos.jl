"""
Empirical Bayes priors that are used in the simulations of:
> Stephens, M., 2017. False discovery rates: a new deal. Biostatistics, 18(2), pp.275-294.
"""
const AshPriors = Dict(
    :Spiky => MixtureModel(
        [
            Normal(0, 0.25)
            Normal(0, 0.5)
            Normal(0, 1)
            Normal(0, 2)
        ],
        [0.4; 0.2; 0.2; 0.2],
    ),
    :NearNormal => MixtureModel([Normal(0, 1); Normal(0, 2)], [1 / 3; 2 / 3]),
    :FlatTop => MixtureModel([Normal(x, 0.5) for x = -1.5:0.5:1.5]),
    :Skew => MixtureModel(
        [Normal(-2, 2), Normal(-1, 1.5), Normal(0, 1), Normal(1, 1)],
        [1 / 4; 1 / 4; 1 / 3; 1 / 6],
    ),
    :BigNormal => Normal(0, 4),
    :Bimodal => MixtureModel([Normal(-2, 1); Normal(2, 1)], [0.5; 0.5]),
)
