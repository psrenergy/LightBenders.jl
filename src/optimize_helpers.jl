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

function treat_termination_status(model::JuMP.Model, t::Integer, s::Integer, file_dir::String)
    if termination_status(model) != MOI.OPTIMAL
        if t == 0 && s == 0
            @info(
                "Deterministic equivalent model finished with termination status: ",
                termination_status(model),
            )
        else
            @info(
                "Model of stage $t, scenario $s finished with termination status: ",
                termination_status(model),
            )
        end
        if termination_status(model) == MOI.INFEASIBLE
            JuMP.compute_conflict!(model)
            if file_dir != "" && !ispath(file_dir)
                mkdir(file_dir)
            end
            file = joinpath(file_dir, "infeasible_model")
            print_conflict_to_file(model, file)
            JuMP.write_to_file(model, string(file, ".lp"))
        end
        error("Optimization failed.")
    end
end