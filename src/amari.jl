#----------------------------------
# Structure
#----------------------------------
# ModulusModel contains all JuMP related information
# AMARI contains optimization information
# SteinMinimaxEstimator contains the fit result
#

abstract type AbstractModulusModel end


Base.@kwdef struct ModulusModelWithF <: AbstractModulusModel
    method
    model
    g1
    g2
    f1
    f2
    f_sqrt
    estimated_marginal_density
    Δf
    δ_max
    δ_up
    bound_delta
    target
end

Base.@kwdef struct ModulusModelWithoutF <: AbstractModulusModel
    method
    model
    g1
    g2
    δ_max
    δ_up
    bound_delta
    target
    discretizer
    representative_eb_samples
end

function Base.show(io::IO, model::AbstractModulusModel)
    println(io, "Modulus model with target:")
    show(io, model.target)
end


function get_δ(Δf)
    norm(JuMP.value.(Δf))
end

# TODO: recalculate δ for general case?
function get_δ(model::AbstractModulusModel)
      JuMP.value(model.δ_up)
end

function get_δ(model::ModulusModelWithF; recalculate_δ = false)
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

function get_bias_var(modulus_model::AbstractModulusModel)
    @unpack model = modulus_model
    δ = get_δ(modulus_model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(modulus_model.bound_delta)
    max_bias = (ω_δ - δ*ω_δ_prime)/2
    unit_var_proxy = ω_δ_prime^2
    max_bias, unit_var_proxy
end

function (bv::BiasVarAggregate)(modulus_model::AbstractModulusModel)
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
    se = sqrt(unit_var_proxy/half_ci.n)
    gaussian_ci(se; maxbias=bias, α=half_ci.α)
end

"""
    AMARI(convexclass::Empirikos.ConvexPriorClass,
          flocalization::Empirikos.FLocalization,
          solver,
          plugin_G = KolmogorovSmirnovMinimumDistance(convexclass, solver))

Affine Minimax Anderson-Rubin intervals for empirical Bayes estimands.
Here `flocalization` is a  pilot [`Empirikos.FLocalization`](@ref), `convexclass` is
a [`Empirikos.ConvexPriorClass`](@ref), `solver` is a JuMP.jl compatible solver.
`plugin_G` is a [`Empirikos.EBayesMethod`](@ref) used as an initial estimate of the marginal
distribution of the i.i.d. samples ``Z``.
"""
Base.@kwdef struct AMARI{N, G, M, EB}
    convexclass::G
    flocalization::N
    solver
    discretizer = nothing#DataBasedDefault()
    plugin_G = KolmogorovSmirnovMinimumDistance(convexclass, solver)
    data_split = :none
    delta_grid = 0.2:0.5:6.7
    delta_objective = RMSE()
    modulus_model::M = ModulusModelWithoutF
    representative_eb_samples::EB = nothing
    n = nothing
end

function initialize_modulus_model(method::AMARI, ::Type{ModulusModelWithF}, target::Empirikos.LinearEBayesTarget, δ)

    #TODO: perhaps this can also be moved..
    @unpack discretizer, solver, convexclass, flocalization, representative_eb_samples = method

    length(representative_eb_samples.vec) == 1 ||
            throw(ArgumentError("ModulusModelWithF only works for Homoskedastic samples."))
    Z = first(representative_eb_samples.vec)
    if isa(method.plugin_G, Distribution) #TODO SPECIAL CASE this elsewhere?
        estimated_marginal_density = Empirikos.dictfun(discretizer, Z, z-> pdf(method.plugin_G, z))
    else
        estimated_marginal_density = Empirikos.dictfun(discretizer, Z, z-> pdf(method.plugin_G.prior, z))
    end

    model = Model(solver)

    g1 = Empirikos.prior_variable!(model, convexclass)
    g2 = Empirikos.prior_variable!(model, convexclass)

    Empirikos.flocalization_constraint!(model, flocalization, g1)
    Empirikos.flocalization_constraint!(model, flocalization, g2)

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

    ModulusModelWithF(method=method, model=model, g1=g1, g2=g2, f1=f1, f2=f2,
        f_sqrt=f_sqrt, estimated_marginal_density=estimated_marginal_density,
        Δf=Δf, δ_max=Inf, δ_up=δ_up,
        bound_delta=bound_delta, target=target)
end

function modulus_cholesky_factor(convexclass::AbstractSimplexPriorClass, plugin_G, discr,
            eb_samples::HeteroskedasticSamples)
    K = nparams(convexclass)
    chr = cholesky(zeros(K, K) + I)
    cache_vec = zeros(K)
    fill!(chr.factors, 0)
    cache_vec  = zeros(K)
    for _interval in discr.sorted_intervals
        for (z, pr) in zip(eb_samples.vec, eb_samples.probs)
            z = set_response(z, _interval)
            cache_vec .= sqrt(pr) .* exp.(logpdf.(components(convexclass), z) .- logpdf(plugin_G, z)/2)
            lowrankupdate!(chr, cache_vec)
        end
    end
    chr
end


function initialize_modulus_model(method::AMARI, ::Type{ModulusModelWithoutF}, target::Empirikos.LinearEBayesTarget, δ)


    @unpack flocalization, convexclass, discretizer, plugin_G, solver, representative_eb_samples = method

    model = Model(solver)

    g1 = Empirikos.prior_variable!(model, convexclass)
    g2 = Empirikos.prior_variable!(model, convexclass)

    Empirikos.flocalization_constraint!(model, flocalization, g1)
    Empirikos.flocalization_constraint!(model, flocalization, g2)

    @variable(model, δ_up)

    K_chol = modulus_cholesky_factor(convexclass, plugin_G, discretizer, representative_eb_samples)
    KΔg = @expression(model, K_chol.U * (g1.finite_param - g2.finite_param))

    @constraint(model, pseudo_chisq_constraint,
           [δ_up; KΔg] in SecondOrderCone())

    @objective(model, Max, target(g2) - target(g1))
    @constraint(model, bound_delta, δ_up <= δ)

    ModulusModelWithoutF(method=method, model=model, g1=g1, g2=g2,
        δ_max=Inf, δ_up=δ_up,
        bound_delta=bound_delta, target=target,
        discretizer = discretizer, representative_eb_samples = representative_eb_samples
        ) #FIXME potentially
end



function set_δ!(modulus_model::AbstractModulusModel, δ)
    set_normalized_rhs(modulus_model.bound_delta, δ)
    optimize!(modulus_model.model)
    modulus_model
end

function set_target!(modulus_model::AbstractModulusModel, target::Empirikos.LinearEBayesTarget)
    #if modulus_model.target == target
    #   return modulus_model
    #end
    @unpack model, g1, g2 = modulus_model
    @objective(model, Max, target(g2) - target(g1))
    modulus_model = @set modulus_model.target = target
    optimize!(model)
    modulus_model
end



function initialize_method(method::AMARI, target::Empirikos.LinearEBayesTarget, Zs; kwargs...)

    method = Empirikos.set_defaults(method, Zs; kwargs...)
    fitted_floc = StatsBase.fit(method.flocalization, Zs; kwargs...)
    if isa(method.plugin_G, Distribution) #TODO SPECIAL CASE this elsewhere?
        fitted_plugin_G = method.plugin_G
    else
        fitted_plugin_G = StatsBase.fit(method.plugin_G, Zs; kwargs...)
    end
    discr = method.discretizer #TODO SPECIAL CASE for ::Distribution
    modulus_model = method.modulus_model
    # todo: fix this
    #if isa(modulus_model, Type{ModulusModelWithF})

    method = @set method.representative_eb_samples = heteroskedastic(Zs)

    method = @set method.flocalization = fitted_floc
    method = @set method.plugin_G = fitted_plugin_G

    n = nobs(Zs) #TODO: length or nobs?
    method = @set method.n = n

    @unpack delta_grid = method

    δ1 = delta_grid[1]
    modulus_model = initialize_modulus_model(method, modulus_model, target, δ1)
    method = @set method.modulus_model = modulus_model
    method
end #AMARI -> AMARI

function StatsBase.fit(method::AMARI, target, Zs; initialize=true, kwargs...)
    if initialize
        method = initialize_method(method, target, Zs; kwargs...)
    end
    _fit_initialized(method::AMARI, target, Zs; kwargs...)
end #AMARI ->




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

Base.@kwdef struct QDonoho{G,H,T,D}
    g1::G
    g2::G
    plugin_G::H
    mult::T
    offset::T
    discretizer::D
end

function (Q::QDonoho)(Z::EBayesSample)
    @unpack g1, g2, plugin_G, mult, offset, discretizer  = Q
    Z = discretizer(Z)
    _g1_val = exp(logpdf(g1, Z) - logpdf(plugin_G, Z))
    _g2_val = exp(logpdf(g2, Z) - logpdf(plugin_G, Z))
    offset + mult*(_g2_val - _g1_val)
end


function SteinMinimaxEstimator(modulus_model::ModulusModelWithoutF)
    @unpack model, method, target, discretizer, representative_eb_samples  = modulus_model
    @unpack convexclass, plugin_G = method

    discretizer == method.discretizer || throw("Internal discretizer modified.")

    δ = get_δ(modulus_model)
    ω_δ = objective_value(model)
    ω_δ_prime = -JuMP.dual(modulus_model.bound_delta)

    g1 = modulus_model.g1()
    g2 = modulus_model.g2()

    L1 = target(g1)
    L2 = target(g2)

    offset_sum = sum(discretizer.sorted_intervals) do _int
        sum(zip(representative_eb_samples.vec, representative_eb_samples.probs)) do (z,pr)
            z = set_response(z, _int)
            f2_z = pdf(g2, z)
            f1_z = pdf(g1, z)
            barf_z = pdf(plugin_G, z)
            pr*(f2_z - f1_z) * (f2_z + f1_z) / barf_z
        end
    end

    offset =  (L1+L2)/2 - ω_δ_prime/(2*δ)*offset_sum
    mult = ω_δ_prime/δ

    Q = QDonoho(;g1 = g1, g2 = g2, plugin_G = plugin_G,
                offset = offset, mult = mult,
                discretizer = discretizer)

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
              modulus_model=modulus_model
        )
end

function SteinMinimaxEstimator(modulus_model::ModulusModelWithF)
    @unpack model, method, target, estimated_marginal_density = modulus_model
    @unpack convexclass = method

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
              modulus_model=modulus_model
        )
end


function _fit_initialized(method::AMARI, target, Zs; kwargs...)
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
    target == Q.modulus_model.target ||
           throw("Target has changed")

    _bias = Q.max_bias
    _Qs = Q.Q.(Zs)
    _wts = StatsBase.weights(Zs)
    _se = std(Q.Q.(Zs), _wts)/sqrt(nobs(Zs))
    point_estimate = mean(Q.Q.(Zs), _wts)
    halfwidth = gaussian_ci(_se; maxbias=_bias, α=α)
    BiasVarianceConfidenceInterval(estimate = point_estimate,
                                   maxbias = _bias,
                                   se = _se,
                                   α = α, method = nothing, target = target)
end

function StatsBase.confint(method::AMARI, target::Empirikos.LinearEBayesTarget, Zs; kwargs...)
    _fit = StatsBase.fit(method, target, Zs; kwargs...)
    StatsBase.confint(_fit, target, Zs; kwargs...)
end

function Base.broadcasted(::typeof(StatsBase.confint), method::AMARI,
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

"""
    StatsBase.confint(method::AMARI,
                      target::Empirikos.EBayesTarget,
                      Zs;
                      α=0.05)

Form a confidence interval for the [`Empirikos.EBayesTarget`](@ref) `target` with coverage
    `1-α` based on the samples `Zs` using the [`AMARI`](@ref) `method`.
"""
function StatsBase.confint(method::AMARI, target::Empirikos.AbstractPosteriorTarget, Zs;
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

    floc_worst_case = FLocalizationInterval(flocalization = method.flocalization,
            convexclass = method.convexclass,
            solver= method.solver)
    outer_ci = StatsBase.confint(floc_worst_case, target)
    outer_ci =  @set outer_ci.α = α

    if isnothing(c_lower)
        c_lower = outer_ci.lower
    end
    if isnothing(c_upper)
        c_upper = outer_ci.upper
    end

    #@show c_lower, c_upper
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

    #@show  confint_lower.lower, confint_lower.upper
    #@show confint_upper.lower, confint_upper.upper
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
    #@show λs_lhs, λs_rhs
    λ_lhs = minimum(λs_lhs)
    λ_rhs = maximum(λs_rhs)


    c_lower_updated = (1-λ_lhs)*c_lower + λ_lhs*c_upper
    c_upper_updated = (1-λ_rhs)*c_lower + λ_rhs*c_upper

    # TODO: Pretruncate according to target

    if c_lower_updated < c_lower || c_upper_updated > c_upper
        return outer_ci
    end
    #let us assume we have flocalizations.

    LowerUpperConfidenceInterval(α=α, target=target, method=nothing,
                                 lower=c_lower_updated,
                                 upper=c_upper_updated)
end


function Base.broadcasted(::typeof(StatsBase.confint), method::AMARI,
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


    confint_vec =  Vector{LowerUpperConfidenceInterval}(undef, length(targets))

    for (index, target) in enumerate(targets)
        _ci = StatsBase.confint(method, target, Zs, args...; initialized=true, kwargs...)
        confint_vec[index] = _ci
    end
    confint_vec
end


# used right now only for sanity check in tests
function worst_case_bias_lp(fitted_amari::AMARI, Q::QDonoho, target; max=true)
    @unpack convexclass, solver, discretizer, flocalization, representative_eb_samples = fitted_amari

    transposed_intervals = reshape(discretizer.sorted_intervals, 1, length(discretizer.sorted_intervals))
    Zs = set_response.(representative_eb_samples.vec, transposed_intervals)
    model = Model(solver)

    g1 = prior_variable!(model, convexclass)
    flocalization_constraint!(model, flocalization, g1)

    Qs = Q.(Zs)
    f_G = @expression(model, pdf.(g1, Zs) .* representative_eb_samples.probs)

    @objective(model, Max, dot(f_G, Qs) - target(g1))
    optimize!(model)
    maxbias = JuMP.objective_value(model)

    @objective(model, Min, dot(f_G, Qs) - target(g1))
    optimize!(model)
    minbias = JuMP.objective_value(model)

    expected_var = dot(Qs.^2, pdf.(fitted_amari.plugin_G, Zs) .* representative_eb_samples.probs)


    (maxbias = maxbias, minbias = minbias, expected_var = expected_var)
end
