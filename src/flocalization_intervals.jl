"""
    FLocalizationInterval(flocalization::Empirikos.FLocalization,
                          convexclass::Empirikos.ConvexPriorClass,
                          solver,
                          n_bisection = 100)

Method for computing frequentist confidence intervals for empirical Bayes
estimands. Here `flocalization` is a  [`Empirikos.FLocalization`](@ref), `convexclass` is
a [`Empirikos.ConvexPriorClass`](@ref), `solver` is a JuMP.jl compatible solver.

`n_bisection` is relevant only for combinations of `flocalization` and `convexclass` for
    which the Charnes-Cooper transformation is not applicable. Instead, a quasi-convex
    optimization problem is solved by bisection and increasing `n_bisection` increases
    accuracy (at the cost of more computation).
"""
Base.@kwdef struct FLocalizationInterval{N,G}
    flocalization::N
    convexclass::G
    solver
    n_bisection::Int = 100
end

function Empirikos.nominal_alpha(floc::FLocalizationInterval)
    Empirikos.nominal_alpha(floc.flocalization)
end


Base.@kwdef struct FittedFLocalizationInterval{T, NW<:FLocalizationInterval, M, P, V}
    method::NW
    target::T = nothing
    model::M
    gmodel::P
    g1::V = nothing
    g2::V = nothing
    lower::Float64 = -Inf
    upper::Float64 = +Inf
end



function StatsBase.fit(method::FLocalizationInterval, target, Zs; kwargs...)
    Zs = Empirikos.summarize_by_default(Zs) ? summarize(Zs) : Zs
    method = Empirikos.set_defaults(method, Zs; kwargs...)

    fitted_floc = StatsBase.fit(method.flocalization, Zs)
    method = @set method.flocalization = fitted_floc

    StatsBase.fit(method, target)
end

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.AbstractPosteriorTarget)
    StatsBase.fit(method, target, Empirikos.vexity(method.flocalization))
end

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.LinearVexity)

    lfp = LinearFractionalModel(method.solver)
    g = Empirikos.prior_variable!(lfp, method.convexclass)

    Empirikos.flocalization_constraint!(lfp, method.flocalization, g)

    fitted_worst_case = FittedFLocalizationInterval(method=method,
        model=lfp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target)
end


function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget},
    target::Empirikos.AbstractPosteriorTarget)
    StatsBase.fit(method, target, Empirikos.vexity(method.method.flocalization))
end

function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget},
                       target::Empirikos.AbstractPosteriorTarget, ::Empirikos.LinearVexity)

    g = method.gmodel
    lfp = method.model

    target_numerator = numerator(target)
    target_numerator_g = target_numerator(g)
    target_denominator = denominator(target)
    target_denominator_g = target_denominator(g)


    set_objective(lfp, JuMP.MOI.MIN_SENSE, target_numerator_g, target_denominator_g)
    optimize!(lfp)
    _min = objective_value(lfp)
    g1 = g()
    set_objective(lfp, JuMP.MOI.MAX_SENSE, target_numerator_g, target_denominator_g)
    optimize!(lfp)
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

function StatsBase.fit(method::FLocalizationInterval{<:Empirikos.FittedFLocalization},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)

    @unpack n_bisection, solver, convexclass, flocalization = method

    _max_vals = Vector{Float64}(undef, n_bisection)
    _min_vals = Vector{Float64}(undef, n_bisection)

    model = Model(solver)
    g = Empirikos.prior_variable!(model, convexclass)

    Empirikos.flocalization_constraint!(model, flocalization, g)
    num_target = numerator(target)
    denom_target = denominator(target)
    @objective(model, Max, denom_target(g))
    optimize!(model)
    _max_denom = JuMP.objective_value(model)
    @objective(model, Min, denom_target(g))
    optimize!(model)
    _min_denom = JuMP.objective_value(model)

    _denom_range = range(_min_denom, stop=_max_denom, length=n_bisection)
    @variable(model, t == _min_denom, Param())
    @constraint(model, denom_target(g) == t)

    for (i, _denom) in enumerate(_denom_range)
        set_value(t, _denom)
        @objective(model, Max, num_target(g))
        optimize!(model)
        _max_vals[i] = target(g())
        @objective(model, Min, num_target(g))
        optimize!(model)
        _min_vals[i] = target(g())
    end

    _max = maximum(_max_vals)
    _min = minimum(_min_vals)

    FittedFLocalizationInterval(method=method,
        target=target,
        model=model,
        gmodel=g,
        g1=nothing,
        g2=nothing,
        lower=_min,
        upper=_max)
end

function StatsBase.fit(method::FittedFLocalizationInterval{<:Empirikos.AbstractPosteriorTarget},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)
    # TODO: Cache results here too? But maybe wait for ParametricOptInterface first
    StatsBase.fit(method.method, target)
end



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
    _min = objective_value(lp)
    g1 = g()

    @objective(lp, Max, target_g)
    optimize!(lp)
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





function StatsBase.confint(fitted_worst_case::FittedFLocalizationInterval)
    @unpack target, method, lower, upper = fitted_worst_case
    α = nominal_alpha(method.flocalization)
    LowerUpperConfidenceInterval(α=α, target=target, lower=lower,
                                 upper=upper)
end

function StatsBase.confint(floc::Union{FLocalizationInterval,FittedFLocalizationInterval}, target, args...)
    _fit = StatsBase.fit(floc, target, args...)
    StatsBase.confint(_fit)
end


function Base.broadcasted(::typeof(StatsBase.confint), floc::FLocalizationInterval, targets, args...)
    _fit = StatsBase.fit(floc, targets[1], args...)
    confint_vec = fill(confint(_fit), axes(targets))
    for (index, target) in enumerate(targets[2:end])
        confint_vec[index+1] = StatsBase.confint(StatsBase.fit(_fit, target))
    end
    confint_vec
end
