function check_moi_optimal(model)
    JuMP.termination_status(model) == MathOptInterface.OPTIMAL ||
    JuMP.termination_status(model) == MathOptInterface.ALMOST_OPTIMAL ||
    JuMP.termination_status(model) == MathOptInterface.SLOW_PROGRESS ||
        throw("status_not_optimal") # TODO: perhaps add warning.
end