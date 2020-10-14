struct NPMLE{C} <: EBayesMethod
    convexclass::C
    solver::Any
    dict::Any
end

NPMLE(convexclass, solver) = NPMLE(convexclass, solver, nothing)

struct FittedNPMLE{D, N<:NPMLE}
    prior::D
    npmle::N
    model::Any # add status?
end
Base.broadcastable(fitted_npmle::FittedNPMLE) = Ref(fitted_npmle)


marginalize(Z, fitted_npmle::FittedNPMLE) = marginalize(Z, fitted_npmle.prior)

function fix_πs(πs)
    πs = max.(πs, 0.0)
    πs = πs ./ sum(πs)
end

function StatsBase.fit(npmle::NPMLE, Zs)
    @unpack convexclass, solver = npmle
    model = Model(solver)

    π = Empirikos.prior_variable!(model, convexclass)
    f = pdf.(π, Zs)

    n = length(f)

    @variable(model, u)

    @constraint(model,  vcat(u, f, fill(1.0,n)) in MathOptInterface.RelativeEntropyCone(2n+1))
    @objective(model, Min, u)
    optimize!(model)
    estimated_prior = convexclass(fix_πs(JuMP.value.(π.finite_param)))
    FittedNPMLE(estimated_prior, npmle, model)
end

# NonparametricMLE( __ optional {ConvexPriorClass}; grid= , ngrid=  , method=:primal or :dual, solver= )

# NPMLE{}

# FModel()

#What are we estimating?
#How are we estimating it?
