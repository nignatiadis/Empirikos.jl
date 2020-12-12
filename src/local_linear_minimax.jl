#----------------------------------
# Structure
#----------------------------------
# ModulusModel contains all JuMP related information
# LocalizedAffineMinimax contains optimization information
# SteinMinimaxEstimator contains the fit result
#





Base.@kwdef struct ModulusModel
    method
    model
    g1
    g2
    f1
    f2
    f_sqrt
    Δf
    δ_max
    δ_up
    bound_delta
    target
end

function get_δ(Δf)
    norm(JuMP.value.(Δf))
end

function get_δ(model::ModulusModel; recalculate_δ = false)
    if recalculate_δ
        get_δ(model.Δf)
    else
      JuMP.value(model.δ_up)
    end
end

"""
    DeltaTuner

Abstract type used to represent ways of picking
``\\delta`` at which to solve the modulus problem, cf.
Manuscript. Different choices of ``\\delta`` correspond
to different choices of the Bias-Variance tradeoff with
every choice leading to Pareto-optimal tradeoff.
"""
abstract type DeltaTuner end

abstract type BiasVarAggregate <: DeltaTuner end

function get_bias_var(modulus_model::ModulusModel)
    @unpack model = modulus_model
    δ = get_δ(modulus_model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(modulus_model.bound_delta)
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2
    max_bias, unit_var_proxy
end

function (bv::BiasVarAggregate)(modulus_model::ModulusModel)
    bv(get_bias_var(modulus_model)...)
end

"""
    RMSE(n::Integer) <: DeltaTuner

A `DeltaTuner` to optimizes
the worst-case (root) mean squared error.  Here `n` is the sample
size used for estimation.
"""
struct RMSE{N} <: BiasVarAggregate
    n::N
end

RMSE() = RMSE(DataBasedDefault())
(rmse::RMSE)(bias, unit_var_proxy) =  sqrt(bias^2 + unit_var_proxy/rmse.n)

function Empirikos._set_defaults(rmse::RMSE, Zs; hints)
    RMSE(nobs(Zs))
end


"""
    HalfCIWidth(n::Integer, α::Float64) <: DeltaTuner

A `DeltaTuner` that chooses the `δ ≧ δ_min` the optimizes
the worst-case confidence interval width.  Here `n` is the sample
size used for estimation.
"""
Base.@kwdef struct HalfCIWidth{N} <: BiasVarAggregate
    n::N = DataBasedDefault()
    α::Float64 = 0.05
end

function Empirikos._set_defaults(halfci::HalfCIWidth, Zs; hints)
    @set halfci.n = nobs(Zs)
end


function (half_ci::HalfCIWidth)(bias, unit_var_proxy)
    @show bias
    se = sqrt(unit_var_proxy/half_ci.n)
    @show se
    @show gaussian_ci(se; maxbias=bias, α=half_ci.α)
end


Base.@kwdef struct LocalizedAffineMinimax{N, G, M}
    convexclass::G
    neighborhood::N
    solver
    discretizer = DataBasedDefault()
    plugin_G
    data_split = :none
    delta_grid = 0.2:0.5:6.7
    delta_objective = RMSE()
    estimated_marginal_density = nothing
    modulus_model::M = nothing
    n = nothing
end

function initialize_modulus_model(method::LocalizedAffineMinimax, target::Empirikos.LinearEBayesTarget, δ)

    estimated_marginal_density = method.estimated_marginal_density
    neighborhood = method.neighborhood
    gcal = method.convexclass

    solver = method.solver

    model = Model(solver)

    g1 = Empirikos.prior_variable!(model, gcal)
    g2 = Empirikos.prior_variable!(model, gcal)

    Empirikos.neighborhood_constraint!(model, neighborhood, g1)
    Empirikos.neighborhood_constraint!(model, neighborhood, g2)

    Zs = collect(keys(estimated_marginal_density))
    f_sqrt = sqrt.(collect(values(estimated_marginal_density)))

    f1 = pdf.(g1, Zs)
    f2 = pdf.(g2, Zs)

    @variable(model, δ_up)

    Δf = @expression(model, (f1 - f2)./f_sqrt)

    @constraint(model, pseudo_chisq_constraint,
           [δ_up; Δf] in SecondOrderCone())

    @objective(model, Max, target(g2) - target(g1))

    @constraint(model, bound_delta, δ_up <= δ)

    ModulusModel(method=method, model=model, g1=g1, g2=g2, f1=f1, f2=f2,
        f_sqrt=f_sqrt, Δf=Δf, δ_max=Inf, δ_up=δ_up,
        bound_delta=bound_delta, target=target)
end


function set_δ!(modulus_model, δ)
    set_normalized_rhs(modulus_model.bound_delta, δ)
    optimize!(modulus_model.model)
    modulus_model
end

function set_target!(modulus_model, target::Empirikos.LinearEBayesTarget)
    #if modulus_model.target == target
    #   return modulus_model
    #end
    @unpack model, g1, g2 = modulus_model
    @objective(model, Max, target(g2) - target(g1))
    modulus_model = @set modulus_model.target = target
    optimize!(model)
    modulus_model
end



function initialize_method(method::LocalizedAffineMinimax, target::Empirikos.LinearEBayesTarget, Zs; kwargs...)

    method = Empirikos.set_defaults(method, Zs; kwargs...)
    fitted_nbhood = StatsBase.fit(method.neighborhood, Zs; kwargs...)
    if isa(method.plugin_G, Distribution) #TODO SPECIAL CASE this elsewhere?
        fitted_plugin_G = method.plugin_G
    else
        fitted_plugin_G = StatsBase.fit(method.plugin_G, Zs; kwargs...)
    end
    discr = method.discretizer #TODO SPECIAL CASE for ::Distribution

    # todo: fix this
    fitted_density = Empirikos.dictfun(discr, Zs, z-> pdf(fitted_plugin_G.prior, z))

    method = @set method.neighborhood = fitted_nbhood
    method = @set method.plugin_G = fitted_plugin_G
    method = @set method.estimated_marginal_density = fitted_density

    n = nobs(Zs) #TODO: length or nobs?
    method = @set method.n = n

    @unpack delta_grid = method

    δ1 = delta_grid[1]
    modulus_model = initialize_modulus_model(method, target, δ1)
    method = @set method.modulus_model = modulus_model
    method
end #LocalizedAffineMinimax -> LocalizedAffineMinimax

function StatsBase.fit(method::LocalizedAffineMinimax, target, Zs; initialize=true, kwargs...)
    if initialize
        method = initialize_method(method, target, Zs; kwargs...)
    end
    _fit_initialized(method::LocalizedAffineMinimax, target, Zs; kwargs...)
end #LocalizedAffineMinimax ->




Base.@kwdef mutable struct SteinMinimaxEstimator{M, T, D}
    target::T
    δ::Float64
    ω_δ::Float64
    ω_δ_prime::Float64
    g1
    g2
    Q::D
    max_bias::Float64
    unit_var_proxy::Float64 #n::Int64
    modulus_model::M #    δ_tuner::DeltaTuner
    method
    δs = zeros(Float64,5)
    δs_objective = zeros(Float64, length(δs))
end


#var(sme::SteinMinimaxEstimator) = sme.unit_var_proxy/sme.n


function set_target!(method::SteinMinimaxEstimator, target)
    @unpack modulus_model = method
    if method.target == target
        return method
    end
    modulus_model = set_target!(modulus_model, target)
    @set method.modulus_model = modulus_model
    @set method.target = target
    method
end

function worst_case_bias(sme::SteinMinimaxEstimator)
    sme.max_bias
end

#"""
#SteinMinimaxEstimator(Zs_discr::DiscretizedStandardNormalSamples,
#              prior_class::ConvexPriorClass,
##              target::LinearEBayesTarget,
#              δ_tuner::DeltaTuner
#
#Computes a linear estimator optimizing a worst-case bias-variance tradeoff (specified by `δ_tuner`)
#for estimating a linear `target` over `prior_class` based on [`DiscretizedStandardNormalSamples`](@ref)
#`Zs_discr`.



function SteinMinimaxEstimator(modulus_model::ModulusModel)
    @unpack model, method, target = modulus_model
    @unpack convexclass, estimated_marginal_density = method

    δ = get_δ(modulus_model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(modulus_model.bound_delta)

    g1 = modulus_model.g1()
    g2 = modulus_model.g2()

    L1 = target(g1)
    L2 = target(g2)

    Zs = collect(keys(estimated_marginal_density))

    f1 = pdf.(g1, Zs)
    f2 = pdf.(g2, Zs)

    f̄s = collect(values(estimated_marginal_density))

    Q = ω_δ_prime/δ*(f2 .- f1)./f̄s
    Q_0  = (L1+L2)/2 -
        ω_δ_prime/(2*δ)*sum( (f2 .- f1).* (f2 .+ f1) ./ f̄s)

    Q = Empirikos.dictfun(method.discretizer, Zs, Q .+ Q_0)

    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2


SteinMinimaxEstimator(
              target=target,
              δ=δ,
              ω_δ=ω_δ,
              ω_δ_prime=ω_δ_prime,
              g1=g1,
              g2=g2,
              Q=Q,
              max_bias=max_bias,
              unit_var_proxy=unit_var_proxy,
              method=method,
              modulus_model=modulus_model)
end


function _fit_initialized(method::LocalizedAffineMinimax, target, Zs; kwargs...)
    @unpack modulus_model, delta_grid, delta_objective, n = method
    modulus_model = set_target!(modulus_model, target) #TODO: This is mutating the JuMP.model but not anything else

    δs_objective = zeros(Float64, length(delta_grid))

    for (index, δ) in enumerate(delta_grid)
        set_δ!(modulus_model, δ/sqrt(n)) #TODO: sanity checks
        δs_objective[index] = delta_objective(modulus_model)
    end

    if length(delta_grid) > 1 # resolve to get best objective
        idx_best = argmin(δs_objective)
        set_δ!(modulus_model, delta_grid[idx_best]/sqrt(n)) #TODO: sanity checks
    end

    sm = SteinMinimaxEstimator(modulus_model)
    sm = @set sm.δs = collect(delta_grid)
    sm = @set sm.δs_objective = δs_objective
    sm = @set sm.method = method
    sm
end


function StatsBase.confint(Q::SteinMinimaxEstimator, target, Zs; α=0.05)
    _bias = Q.max_bias
    _Qs = Q.Q.(Zs)
    _wts = StatsBase.weights(Zs)
    _se = std(Q.Q.(Zs), _wts)/sqrt(nobs(Zs))
    point_estimate = mean(Q.Q.(Zs), _wts)
    halfwidth = MinimaxCalibratedEBayes.gaussian_ci(_se; maxbias=_bias, α=α)
    BiasVarianceConfidenceInterval(estimate = point_estimate,
                                   maxbias = _bias,
                                   se = _se,
                                   α = α, method = Q.method, target = target)
end

function StatsBase.confint(method::LocalizedAffineMinimax, target::Empirikos.LinearEBayesTarget, Zs; kwargs...)
    _fit = StatsBase.fit(method, target, Zs; kwargs...)
    StatsBase.confint(_fit, target, Zs; kwargs...)
end

function Base.broadcasted(::typeof(StatsBase.confint), method::LocalizedAffineMinimax,
                           targets::AbstractVector{<:Empirikos.LinearEBayesTarget}, Zs, args...; kwargs...)
    length(targets) >= 2  || throw(error("use non-broadcasting call to .fit"))
    mid_idx = ceil(Int,median(Base.OneTo(length(targets))))
    _fit = StatsBase.fit(method, targets[mid_idx], Zs, args...; kwargs...)
    _confint = StatsBase.confint(_fit, targets[mid_idx], Zs, args...; kwargs...)
    confint_vec = fill(_confint, length(targets))

    #TODO make this interface point nicer
    #-------------------------------------
    updated_method = _fit.method
    δ = _fit.δ
    updated_method = @set updated_method.delta_grid = [δ]
    #-------------------------------------

    for (index, target) in enumerate(targets)
        _fit = _fit_initialized(updated_method, target, Zs, args...; kwargs...)
        confint_vec[index] = StatsBase.confint(_fit, target, Zs, args...; kwargs...)
    end
    confint_vec
end


function StatsBase.confint(method::LocalizedAffineMinimax, target::Empirikos.AbstractPosteriorTarget, Zs;
                          initialized=false, α=0.05, c_lower=nothing, c_upper=nothing, single_delta=false, kwargs...)
    if !initialized
        init_target = Empirikos.PosteriorTargetNullHypothesis(target, 0.0)
        method = initialize_method(method, init_target, Zs; kwargs...)

        init_target = Empirikos.PosteriorTargetNullHypothesis(target, target(method.plugin_G))
        _fit = _fit_initialized(method, init_target, Zs)

        if single_delta
            δ = _fit.δ
            method = @set method.delta_grid = [δ]
        end
    end

    nbhood_worst_case = NeighborhoodWorstCase(neighborhood = method.neighborhood,
            convexclass = method.convexclass,
            solver= method.solver)
    outer_ci = StatsBase.confint(nbhood_worst_case, target)
    outer_ci =  @set outer_ci.α = α

    if isnothing(c_lower)
        @show c_lower = outer_ci.lower
    end
    if isnothing(c_upper)
        @show c_upper = outer_ci.upper
    end

    target_lower = Empirikos.PosteriorTargetNullHypothesis(target, c_lower)
    target_upper = Empirikos.PosteriorTargetNullHypothesis(target, c_upper)

    n = nobs(Zs)

    _fit = _fit_initialized(method, target_lower, Zs) #SteinMinimax



    Q_lower = _fit.Q.(Zs)
    _wts = StatsBase.weights(Zs)
    confint_lower = StatsBase.confint(_fit, target_lower, Zs; α=α)
    max_bias_lower = confint_lower.maxbias   # TODO: extract from better CI object
    var_Q_lower = var(Q_lower, _wts)/n
    estimate_lower = confint_lower.estimate

    _fit = _fit_initialized(method, target_upper, Zs) #TODO: should have some bang!?
    Q_upper = _fit.Q.(Zs)
    confint_upper = StatsBase.confint(_fit, target_upper, Zs ; α=α)
    max_bias_upper = confint_upper.maxbias
    var_Q_upper = var(Q_upper, _wts)/n
    estimate_upper = confint_upper.estimate

    cov_lower_upper = cov([Q_lower Q_upper], _wts)[1,2]/n

    @show  confint_lower.lower, confint_lower.upper
    @show confint_upper.lower, confint_upper.upper
    if confint_lower.lower <= 0.0 || confint_upper.upper >= 0.0
        return outer_ci
    end

    bisection_pair = BisectionPair(c1 = c_lower, var1 = var_Q_lower, max_bias1= max_bias_lower, estimate1= estimate_lower,
                         c2 = c_upper, var2 = var_Q_upper, max_bias2= max_bias_upper, estimate2= estimate_upper,
                         cov=cov_lower_upper)

    # TODO: move all functionality here into separate fit function and export as plot
    #λs=0.0:0.001:1.0
    #all_cis = confint.(tmp_pair, λs)
    #all_cis_lower = first.(all_cis)
    #all_cis_upper = last.(all_cis)
    #plot(λs, [all_cis_lower all_cis_upper])


    λs_lhs =  find_zeros(λ -> first(confint(bisection_pair, λ; α=α))  , 0.0, 1.0)
    λs_rhs =  find_zeros(λ -> last(confint(bisection_pair, λ; α=α))  , 0.0, 1.0)
    @show λs_lhs, λs_rhs
    λ_lhs = minimum(λs_lhs)
    λ_rhs = maximum(λs_rhs)


    c_lower_updated = (1-λ_lhs)*c_lower + λ_lhs*c_upper
    c_upper_updated = (1-λ_rhs)*c_lower + λ_rhs*c_upper

    # TODO: Pretruncate according to target

    if c_lower_updated < c_lower || c_upper_updated > c_upper
        return outer_ci
    end
    #let us assume we have neighborhoods.

    LowerUpperConfidenceInterval(α=α, target=target, method=method,
                                 lower=c_lower_updated,
                                 upper=c_upper_updated)
end


function Base.broadcasted(::typeof(StatsBase.confint), method::LocalizedAffineMinimax,
            targets::AbstractVector{<:Empirikos.AbstractPosteriorTarget}, Zs, args...; kwargs...)

    length(targets) >= 2  || throw(error("use non-broadcasting call to .fit"))
    mid_idx = ceil(Int,median(Base.OneTo(length(targets))))

    target = targets[mid_idx]
    init_target = Empirikos.PosteriorTargetNullHypothesis(target, 0.0)
    method = initialize_method(method, init_target, Zs; kwargs...)

    init_target = Empirikos.PosteriorTargetNullHypothesis(target, target(method.plugin_G))
    _fit = _fit_initialized(method, init_target, Zs)

    single_delta = false
    if single_delta
        δ = _fit.δ
        method = @set method.delta_grid = [δ]
    end


    confint_vec =  Vector{MinimaxCalibratedEBayes.LowerUpperConfidenceInterval}(undef, length(targets))

    for (index, target) in enumerate(targets)
        _ci = StatsBase.confint(method, target, Zs, args...; initialized=true, kwargs...)
        confint_vec[index] = _ci
    end
    confint_vec
end
