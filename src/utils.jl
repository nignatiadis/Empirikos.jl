function check_moi_optimal(model)
    status = JuMP.termination_status(model)
    
    if status == MathOptInterface.SLOW_PROGRESS
        @warn "Solver terminated with SLOW_PROGRESS status"
    end
    
    status == MathOptInterface.OPTIMAL ||
    status == MathOptInterface.ALMOST_OPTIMAL ||
    status == MathOptInterface.SLOW_PROGRESS ||
        throw("Status not optimal: $(status)")
end