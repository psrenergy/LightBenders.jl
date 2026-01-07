function save_deterministic_results(
    model::JuMP.Model,
    num_scenarios::Int,
)
    # Get the pre-built variable mapping (avoids string parsing)
    var_scenario_map = model.ext[:var_scenario_map]

    # Get state variables
    state_var_set = Set(model.ext[:first_stage_state].variables)

    # Pre-allocate results dictionary
    results = Dict{Tuple{String, Int}, Any}()

    # Single pass through all variables - use the pre-built mapping
    all_vars = JuMP.all_variables(model)

    for var in all_vars
        value = JuMP.value(var)

        if var in state_var_set
            # State variable - replicate across all scenarios
            name = JuMP.name(var)
            if isempty(name)
                continue  # Skip unnamed variables
            end

            for scen in 1:num_scenarios
                results[(name, scen)] = value
            end
        elseif haskey(var_scenario_map, var)
            # Scenario-specific variable - use pre-built mapping
            (name, scenario) = var_scenario_map[var]
            results[(name, scenario)] = value
        end
        # Skip CVaR auxiliary variables and other unnamed variables
    end

    return results
end
