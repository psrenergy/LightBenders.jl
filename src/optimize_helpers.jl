"""
    SubproblemModel(model::JuMP.Model)

A constructor of JuMP Models with an explicit place to all caches that must exist in the model.
"""
function SubproblemModel(model::JuMP.Model)
    model.ext[:first_stage_state] = StateCache()
    model.ext[:second_stage_state] = StateCache()
    return model
end

function print_conflict_to_file(model::JuMP.Model, filename::String = "infeasible_model")
    if get_attribute(model, MOI.ConflictStatus()) == MOI.CONFLICT_FOUND
        open(filename * ".iis.txt", "w") do io
            write(io, "IIS found\n")
            for cref in all_constraints(model, include_variable_in_set_constraints = true)
                if MOI.get(model, MOI.ConstraintConflictStatus(), cref) == MOI.IN_CONFLICT
                    println(io, cref)
                end
            end
        end
    end
    return nothing
end

function treat_termination_status(model::JuMP.Model, options::DeterministicEquivalentOptions)
    file_name = "det_eq_model"
    info_msg = "Deterministic equivalent model finished with termination status: $(termination_status(model))"
    treat_termination_status(model, info_msg, file_name, options.debugging_options)
    return nothing
end

function treat_termination_status(model::JuMP.Model, options::PolicyTrainingOptions, t::Int, iter::Int)
    file_name = "model_stage_$(t)_iteration_$(iter)"
    info_msg = "Training model of stage $t, iteration $iter finished with termination status: $(termination_status(model))"
    treat_termination_status(model, info_msg, file_name, options.debugging_options)
    return nothing
end

function treat_termination_status(model::JuMP.Model, options::PolicyTrainingOptions, t::Int, s::Int, iter::Int)
    file_name = "model_stage_$(t)_scenario_$(s)_iteration_$(iter)"
    info_msg = "Training model of stage $t, scenario $s, iteration $iter finished with termination status: $(termination_status(model))"
    treat_termination_status(model, info_msg, file_name, options.debugging_options)
    return nothing
end

function treat_termination_status(model::JuMP.Model, options::SimulationOptions)
    file_name = "model_first_stage"
    info_msg = "Simulation model (first stage), finished with termination status: $(termination_status(model))"
    treat_termination_status(model, info_msg, file_name, options.debugging_options)
    return nothing
end

function treat_termination_status(model::JuMP.Model, options::SimulationOptions, t::Int, s::Int)
    file_name = "model_stage_$(t)_scenario_$(s)"
    info_msg = "Simulation model stage $t, scenario $s finished with termination status: $(termination_status(model))"
    treat_termination_status(model, info_msg, file_name, options.debugging_options)
    return nothing
end

function treat_termination_status(model::JuMP.Model, info_msg::String, file_name::String, debugging_options::DebuggingOptions)
    file_name
    infeasible_file_name = string("infeasible_", file_name)
    logs_dir = debugging_options.logs_dir
    if debugging_options.write_lp
        treat_logs_dir(logs_dir)
        JuMP.write_to_file(model, joinpath(logs_dir, string(file_name, ".lp")))
    end
    if termination_status(model) != MOI.OPTIMAL
        @info(info_msg)
        if termination_status(model) == MOI.INFEASIBLE
            treat_logs_dir(logs_dir)
            JuMP.compute_conflict!(model)
            file_path = joinpath(logs_dir, infeasible_file_name)
            print_conflict_to_file(model, file_path)
            JuMP.write_to_file(model, string(file_path, ".lp"))
            error("Optimization failed.")
        end
    end
    return nothing
end

function treat_logs_dir(logs_dir::String)
    if logs_dir != "" && !ispath(logs_dir)
        mkdir(logs_dir)
    end
    return nothing
end
