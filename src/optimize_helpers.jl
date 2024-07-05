"""
    SubproblemModel(model::JuMP.Model)

A constructor of JuMP Models with an explicit place to all caches that must exist in the model.
"""
function SubproblemModel(model::JuMP.Model)
    model.ext[:state] = StateCache()
    return model
end

function print_conflict_to_file(model::JuMP.Model, filename::String = "infeasible_model")
    if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
        open(filename * ".iis.txt", "w") do io
            write(io, "IIS found\n")
            for cref in all_constraints(model, include_variable_in_set_constraints = true)
                if MOI.get(model, MOI.ConstraintConflictStatus(), cref) == MOI.IN_CONFLICT
                    println(io,  cref)
                end
            end
        end
    end
    return nothing
end

function treat_termination_status(model::JuMP.Model, t::Integer, s::Integer)
    if termination_status(model) != MOI.OPTIMAL
        @info(
            "Model of stage $t, scenario $s finished with termination status: ", termination_status(model)
        )
        if termination_status(model) == MOI.INFEASIBLE
            JuMP.compute_conflict!(model)
            print_conflict_to_file(model)
        end
        error("Optimization failed.")
    end
end