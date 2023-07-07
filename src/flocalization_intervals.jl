function check_moi_optimal(model)
    JuMP.termination_status(model) == MathOptInterface.OPTIMAL ||
    JuMP.termination_status(model) == MathOptInterface.ALMOST_OPTIMAL ||
        throw("status_not_optimal") # TODO: perhaps add warning.
end

"""
    FLocalizationInterval(flocalization::Empirikos.FLocalization,
                          convexclass::Empirikos.ConvexPriorClass,
                          solver,
                          n_bisection = 100)

Method for computing frequentist confidence intervals for empirical Bayes
estimands. Here `flocalization` is a  [`Empirikos.FLocalization`](@ref), `convexclass` is
a [`Empirikos.ConvexPriorClass`](@ref), `solver` is a JuMP.jl compatible solver.

`n_bisection` is relevant only for combinations of `target`, `flocalization`
    and `convexclass` for which the Charnes-Cooper transformation
    is not applicable/implemented.
    Instead, a quasi-convex optimization problem is solved by bisection and
    increasing `n_bisection` increases
    accuracy (at the cost of more computation).
"""
Base.@kwdef struct FLocalizationInterval{N,G}
    flocalization::N
    convexclass::G
    solver
    n_bisection::Int = 100
end

function Base.show(io::IO, floc::FLocalizationInterval)
    print(io, "EB intervals with F-Localization: ")
    show(io, floc.flocalization)
    print(io, "\n")
    print(io, "                  𝒢: ")
    show(io, floc.convexclass)
end

function Empirikos.nominal_alpha(floc::FLocalizationInterval)
    Empirikos.nominal_alpha(floc.flocalization)
end


Base.@kwdef struct FittedFLocalizationInterval{T, NW<:FLocalizationInterval, M, P, V}
    method::NW
    target::T = nothing
    model::M  #JuMP model
    gmodel::P #prior variable
    g1::V = nothing #prior corresponding to smallest value of target
    g2::V = nothing #prior corresponding to largest value of target
    lower::Float64 = -Inf
    upper::Float64 = +Inf
end

function Empirikos.nominal_alpha(floc::FittedFLocalizationInterval)
    Empirikos.nominal_alpha(floc.method)
end

# Calls:

## Level 1: With raw data Zs, Need to fit FLocalization
#--------------------------------------------------------
# fit(method::FLocalizationInterval, target, Zs, args...; kwargs...)

## Level 2: Dispatch with FittedFLocalization, maybe this should be renamed initialize or sth like that
#-------------------------------------------------------------------------------------------------------
# fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization}, target::Empirikos.AbstractPosteriorTarget)
# fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization}, target::Empirikos.AbstractPosteriorTarget, vexity::Empirikos.LinearVexity)
# fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization}, target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)
# fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization}, target::Empirikos.LinearEBayesTarget)

## Level 3: Dispatch with FittedFLocalization & JuMP objects and so forth already setup (FittedFLocalizationInterval)
#------------------------------------------------------------------------------------------------------------------------
# fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget}, target::Empirikos.AbstractPosteriorTarget)
# fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget}, target::Empirikos.AbstractPosteriorTarget, ::Empirikos.LinearVexity)
# it(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget}, target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)

# fit(method::FittedFLocalizationInterval{<:Empirikos.LinearEBayesTarget}, target::Empirikos.LinearEBayesTarget)


function StatsBase.fit(method::FLocalizationInterval, target, Zs, args...; kwargs...)
    Zs = Empirikos.summarize_by_default(Zs) ? summarize(Zs) : Zs
    method = Empirikos.set_defaults(method, Zs; kwargs...)

    fitted_floc = StatsBase.fit(method.flocalization, Zs)
    method = @set method.flocalization = fitted_floc

    #init_fitted_floc = initialize_fitted_floc(method, target)
    StatsBase.fit(method, target, args...) #args could be vexity
end

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.AbstractPosteriorTarget)
    StatsBase.fit(method, target, Empirikos.vexity(method.flocalization))
end

function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget},
    target::Empirikos.AbstractPosteriorTarget)
    StatsBase.fit(method, target, Empirikos.vexity(method.method.flocalization))
end

#--------------------------------------------------------------------------
# PosteriorTarget, LinearVexity
#--------------------------------------------------------------------------

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.AbstractPosteriorTarget, vexity::Empirikos.LinearVexity)

    lfp = LinearFractionalModel(method.solver)
    g = Empirikos.prior_variable!(lfp, method.convexclass)

    Empirikos.flocalization_constraint!(lfp, method.flocalization, g)

    fitted_worst_case = FittedFLocalizationInterval(method=method,
        model=lfp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target, vexity)
end

function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget},
                       target::Empirikos.AbstractPosteriorTarget, ::Empirikos.LinearVexity)

                       #TODO CHECK target == method.target ?
    g = method.gmodel
    lfp = method.model

    target_numerator_g = numerator(target)(g)
    target_denominator_g = denominator(target)(g)

    set_objective(lfp, JuMP.MOI.MIN_SENSE, target_numerator_g, target_denominator_g)
    optimize!(lfp)
    check_moi_optimal(lfp)
    _min = objective_value(lfp)

    g1 = g()
    set_objective(lfp, JuMP.MOI.MAX_SENSE, target_numerator_g, target_denominator_g)
    optimize!(lfp)
    check_moi_optimal(lfp)
    _max = objective_value(lfp)
    g2 = g()

    FittedFLocalizationInterval(method=method.method,
        target=target,
        model=lfp,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end

#--------------------------------------------------------------------------
# PosteriorTarget, ConvexVexity
#--------------------------------------------------------------------------


function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)

    @unpack n_bisection, solver, convexclass, flocalization = method

    _max_vals = Vector{Float64}(undef, n_bisection)
    _min_vals = Vector{Float64}(undef, n_bisection)

    model = Model(solver)
    g = Empirikos.prior_variable!(model, convexclass)

    Empirikos.flocalization_constraint!(model, flocalization, g)
    num_target_g = numerator(target)(g)
    denom_target_g = denominator(target)(g)

    @objective(model, Max, denom_target_g)
    optimize!(model)
    check_moi_optimal(model)
    _max_denom = JuMP.objective_value(model)

    @objective(model, Min, denom_target_g)
    optimize!(model)
    check_moi_optimal(model)
    _min_denom = JuMP.objective_value(model)

    #@show _min_denom, _max_denom
    #_diff_denom = _max_denom - _min_denom
    #if _diff_denom > 0
    #    _min_denom = _min_denom + _diff_denom/n_bisection/10
    #    _max_denom = _max_denom - _diff_denom/n_bisection/10
    #end
    #@show _min_denom, _max_denom

    _denom_range = range(_min_denom, stop=_max_denom, length=n_bisection)

    @variable(model, t == _min_denom, Param())
    @constraint(model, denom_target_g == t)

    for (i, _denom) in enumerate(_denom_range)
        set_value(t, _denom)
        @objective(model, Max, num_target_g)
        optimize!(model)
        check_moi_optimal(model)

        _max_vals[i] = target(g())
        @objective(model, Min, num_target_g)
        optimize!(model)
        check_moi_optimal(model)
        _min_vals[i] = target(g())
    end

    _max, _max_idx = findmax(_max_vals)
    _min, _min_idx = findmin(_min_vals)

    # get g2
    set_value(t, _denom_range[_max_idx])
    @objective(model, Max, num_target_g)
    optimize!(model)
    g2 = g()

    # get g1
    set_value(t, _denom_range[_min_idx])
    @objective(model, Min, num_target_g)
    optimize!(model)
    g1 = g()





    FittedFLocalizationInterval(method=method,
        target=target,
        model=model,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end

function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget},
    target::Empirikos.AbstractPosteriorTarget, vexity::Empirikos.ConvexVexity)
    # TODO: Cache results here too? But maybe wait for ParametricOptInterface first
    # Right now we just extract the FLocalizationInterval object and refit the whole
    # thing, including regenerating JuMP models and so forth.
    StatsBase.fit(method.method, target, vexity)
end


#--------------------------------------------------------------------------
# PosteriorVariance
#--------------------------------------------------------------------------

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.PosteriorVariance)

    @unpack n_bisection = method

    postmean_target = PosteriorMean(location(target))

    _fit = StatsBase.fit(method, postmean_target)

    min_postmean = _fit.lower
    max_postmean = _fit.upper

    postmean_range = range(min_postmean, stop=max_postmean, length=n_bisection)
    second_moment_targets = PosteriorSecondMoment.(location(target), postmean_range)

    _tmp_confints = confint.(method, second_moment_targets)

    idx = argmin(getfield.(_tmp_confints, :lower))

    _fit = StatsBase.fit(method, second_moment_targets[idx])
    _fit = @set _fit.target = target
    _fit
end


function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.PosteriorVariance},
    target::Empirikos.PosteriorVariance)
    # TODO: Cache results here too
    StatsBase.fit(method.method, target)
end


#---------------------
# LinearEbayesTarget
#---------------------

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.LinearEBayesTarget)

    lp = Model(method.solver)
    g = Empirikos.prior_variable!(lp, method.convexclass)

    Empirikos.flocalization_constraint!(lp, method.flocalization, g)

    fitted_worst_case = FittedFLocalizationInterval(method=method,
        model=lp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target)
end


function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.LinearEBayesTarget},
                       target::Empirikos.LinearEBayesTarget)

    g = method.gmodel
    lp = method.model

    target_g = target(g)

    @objective(lp, Min, target_g)
    optimize!(lp)
    check_moi_optimal(lp)
    _min = objective_value(lp)
    g1 = g()

    @objective(lp, Max, target_g)
    optimize!(lp)
    check_moi_optimal(lp)
    _max = objective_value(lp)
    g2 = g()

    FittedFLocalizationInterval(method=method.method,
        target=target,
        model=lp,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end

#------------------------------------------------------------------------
# Convenience functions for extracting/computing confidence intervals
#------------------------------------------------------------------------


function confint(
    fitted_worst_case::FittedFLocalizationInterval;
    level=1-nominal_alpha(fitted_worst_case)
)
    @unpack target, method, lower, upper = fitted_worst_case
    α = nominal_alpha(fitted_worst_case)
    α ≈ 1-level || error("F-Localization is not at correct confidence-level")
    LowerUpperConfidenceInterval(α=α, target=target, lower=lower,
                                 upper=upper)
end

function confint(
    floc::Union{FLocalizationInterval,FittedFLocalizationInterval},
    target, args...;
    level=1-nominal_alpha(floc)
)
    # could potentially think about having this reset the confidence level of the FLOC
    # for now be safe and throw an error if wrong!
    Empirikos.nominal_alpha(floc) ≈ 1-level ||
        error("F-Localization is not at correct confidence-level")

    _fit = StatsBase.fit(floc, target, args...)
    confint(_fit; level=level)
end


function Base.broadcasted(::typeof(confint), floc::FLocalizationInterval, targets, args...)
    _fit = StatsBase.fit(floc, targets[1], args...)
    confint_vec = fill(confint(_fit), axes(targets))
    for (index, target) in enumerate(targets[2:end])
        # TODO: Right now this would not work if args... specified non-default vexity.
        confint_vec[index+1] = confint(StatsBase.fit(_fit, target))
    end
    confint_vec
end


function Base.broadcasted_kwsyntax(
    ::typeof(confint),
    floc::FLocalizationInterval,
    targets::AbstractArray,
    args...;
    level=0.95,
    kwargs...,
)
    _fit = StatsBase.fit(floc, targets[1], args...)
    _first_ci = confint(_fit; level=level, kwargs...)
    confint_vec = fill(_first_ci, axes(targets))
    for (index, target) in enumerate(targets[2:end])
            # TODO: Right now this would not work if args... specified non-default vexity.
        confint_vec[index + 1] = confint(_fit, target; level=level)
    end
    return confint_vec
end
