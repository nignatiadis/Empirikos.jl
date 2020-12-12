Base.@kwdef struct NeighborhoodWorstCase{N,G}
    neighborhood::N
    convexclass::G
    solver
    n_bisection::Int = 100
end

function Empirikos.nominal_alpha(nbhood::NeighborhoodWorstCase)
    Empirikos.nominal_alpha(nbhood.neighborhood)
end


Base.@kwdef struct FittedNeighborhoodWorstCase{T, NW<:NeighborhoodWorstCase, M, P, V}
    method::NW
    target::T = nothing
    model::M
    gmodel::P
    g1::V = nothing
    g2::V = nothing
    lower::Float64 = -Inf
    upper::Float64 = +Inf
end



function StatsBase.fit(method::NeighborhoodWorstCase, target, Zs; kwargs...)
    Zs = Empirikos.summarize_by_default(Zs) ? summarize(Zs) : Zs
    method = Empirikos.set_defaults(method, Zs; kwargs...)

    fitted_nbhood = StatsBase.fit(method.neighborhood, Zs)
    method = @set method.neighborhood = fitted_nbhood

    StatsBase.fit(method, target)
end

function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    target::Empirikos.AbstractPosteriorTarget)
    StatsBase.fit(method, target, Empirikos.vexity(method.neighborhood))
end

function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.LinearVexity)

    lfp = LinearFractionalModel(method.solver)
    g = Empirikos.prior_variable!(lfp, method.convexclass)

    Empirikos.neighborhood_constraint!(lfp, method.neighborhood, g)

    fitted_worst_case = FittedNeighborhoodWorstCase(method=method,
        model=lfp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target)
end


function StatsBase.fit(method::FittedNeighborhoodWorstCase{<:Empirikos.AbstractPosteriorTarget},
    target::Empirikos.AbstractPosteriorTarget)
    StatsBase.fit(method, target, Empirikos.vexity(method.method.neighborhood))
end

function StatsBase.fit(method::FittedNeighborhoodWorstCase{<:Empirikos.AbstractPosteriorTarget},
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

    FittedNeighborhoodWorstCase(method=method.method,
        target=target,
        model=lfp,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end

function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)

    @unpack n_bisection, solver, convexclass, neighborhood = method

    _max_vals = Vector{Float64}(undef, n_bisection)
    _min_vals = Vector{Float64}(undef, n_bisection)

    model = ModelWithParams(solver)
    g = Empirikos.prior_variable!(model, convexclass)

    Empirikos.neighborhood_constraint!(model, neighborhood, g)
    num_target = numerator(target)
    denom_target = denominator(target)
    @objective(model, Max, denom_target(g))
    optimize!(model)
    _max_denom = JuMP.objective_value(model)
    @objective(model, Min, denom_target(g))
    optimize!(model)
    _min_denom = JuMP.objective_value(model)

    _denom_range = range(_min_denom, stop=_max_denom, length=n_bisection)
    t = add_parameter(model, _min_denom)
    @constraint(model, denom_target(g) == t)

    for (i, _denom) in enumerate(_denom_range)
        fix(t, _denom)
        @objective(model, Max, num_target(g))
        optimize!(model)
        _max_vals[i] = target(g())
        @objective(model, Min, num_target(g))
        optimize!(model)
        _min_vals[i] = target(g())
    end

    _max = maximum(_max_vals)
    _min = minimum(_min_vals)

    FittedNeighborhoodWorstCase(method=method,
        target=target,
        model=model,
        gmodel=g,
        g1=nothing,
        g2=nothing,
        lower=_min,
        upper=_max)
end

function StatsBase.fit(method::FittedNeighborhoodWorstCase{<:Empirikos.AbstractPosteriorTarget},
    target::Empirikos.AbstractPosteriorTarget, ::Empirikos.ConvexVexity)
    # TODO: Cache results here too? But maybe wait for ParametricOptInterface first
    StatsBase.fit(method.method, target)
end



function StatsBase.fit(method::NeighborhoodWorstCase{<:Empirikos.FittedEBayesNeighborhood},
    target::Empirikos.LinearEBayesTarget)

    lp = Model(method.solver)
    g = Empirikos.prior_variable!(lp, method.convexclass)

    Empirikos.neighborhood_constraint!(lp, method.neighborhood, g)

    fitted_worst_case = FittedNeighborhoodWorstCase(method=method,
        model=lp,
        gmodel=g,
        target=target)

    StatsBase.fit(fitted_worst_case, target)
end

function StatsBase.fit(method::FittedNeighborhoodWorstCase{<:Empirikos.LinearEBayesTarget},
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

    FittedNeighborhoodWorstCase(method=method.method,
        target=target,
        model=lp,
        gmodel=g,
        g1=g1,
        g2=g2,
        lower=_min,
        upper=_max)
end





function StatsBase.confint(fitted_worst_case::FittedNeighborhoodWorstCase)
    @unpack target, method, lower, upper = fitted_worst_case
    α = nominal_alpha(method.neighborhood)
    LowerUpperConfidenceInterval(α=α, target=target, method=method, lower=lower,
                                 upper=upper)
end

function StatsBase.confint(nbhood::Union{NeighborhoodWorstCase,FittedNeighborhoodWorstCase}, target, args...)
    _fit = StatsBase.fit(nbhood, target, args...)
    StatsBase.confint(_fit)
end


function Base.broadcasted(::typeof(StatsBase.confint), nbhood::NeighborhoodWorstCase, targets, args...)
    _fit = StatsBase.fit(nbhood, targets[1], args...)
    confint_vec = fill(confint(_fit), length(targets))
    for (index, target) in enumerate(targets[2:end])
        confint_vec[index+1] = StatsBase.confint(StatsBase.fit(_fit, target))
    end
    confint_vec
end
