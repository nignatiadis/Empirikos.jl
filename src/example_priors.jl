"""
    AshPriors

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

"""
    MarronWandGaussianMixtures

Flexible Gaussian Mixture distributions described in

> Marron, J. Steve, and Matt P. Wand. Exact mean integrated squared error.
The Annals of Statistics (1992): 712-736.
"""
const MarronWandGaussianMixtures = Dict(
    :KurtoticUnimodal => MixtureModel([Normal(0, 1), Normal(0, 0.1)], [2 / 3, 1 / 3]),
    :SkewedUnimodal => MixtureModel(
        [Normal(0, 1), Normal(0.5, 2 / 3), Normal(13 / 12, 5 / 9)],
        [1 / 5, 1 / 5, 3 / 5],
    ),
    :Bimodal => MixtureModel([Normal(-1.0, 2 / 3), Normal(1.0, 2 / 3)]),
    :SeparatedBimodal => MixtureModel([Normal(-1.5, 0.5), Normal(1.5, 0.5)]),
    :SkewedBimodal => MixtureModel([Normal(0, 1), Normal(3 / 2, 1 / 3)], [0.75, 0.25]),
    :Outlier => MixtureModel([Normal(0, 1), Normal(0, 1 / 10)], [1 / 10, 9 / 10]),
)


"""
    IWPriors

Empirical Bayes priors that are used in the simulations of:
> Ignatiadis, N. and Wager, S., 2019. Bias-aware confidence intervals for empirical Bayes analysis. arXiv preprint arXiv:1902.02774.
"""
const IWPriors = Dict(
    :Unimodal => MixtureModel([Normal(-0.2, 0.2), Normal(0, 0.9)], [0.7, 0.3]),
    :Bimodal => MixtureModel([Normal(-1.5, 0.2), Normal(1.5, 0.2)]),
    :NegSpiky => MixtureModel([Normal(-0.25, 0.25), Normal(0, 1.0)], [0.8, 0.2]),
)


"""
    EfronPriors

Empirical Bayes priors that are used in the simulations of:
> Efron, B., 2016. Empirical Bayes deconvolution estimates. Biometrika, 103(1), pp.1-20.
"""
const EfronPriors = Dict(
    :UniformNormal =>
        MixtureModel([Uniform(-3.0, 3.0), Normal(0.0, 0.5)], [1 / 8, 7 / 8]),
)



"""
    CochranePriors
"""
const CochranePriors = Dict(
    :vanZwetSchwabSenn =>
        MixtureModel(Normal.(0, [0.61; 1.42; 2.16; 5.64]), [0.32; 0.31; 0.30; 0.07]),
    :vanZwetGelman => MixtureModel(Normal.(0, [1.5; 3.5]), [0.42; 0.58]),
    :vanZwetGoodman => MixtureModel(
        Normal.([-0.28; -0.22; -0.25; -1.05], [0.78; 1.25; 2.37; 5.85]),
        [0.33; 0.31; 0.30; 0.06],
    ),
)

"""
   OSCPriors
"""
const OSCPriors = Dict(:vanZwetGelman => MixtureModel(Normal.(0, [0.7; 4.0]), [0.57; 0.43]))
