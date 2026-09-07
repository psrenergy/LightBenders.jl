function serial_benders_simulate(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    policy::Policy,
    simulation_options::SimulationOptions,
)
    scenarios = simulation_options.num_scenarios

    simulation_total_cost = 0.0

    results = Dict{Tuple{String, Int}, Any}() # (variable_name, scenario) => value

    # first stage
    if simulation_options.verbose
        @info("Simulating first stage...")
    end

    stage = 1
    state_variables_model = state_variables_builder(inputs, stage)
    model = first_stage_builder(state_variables_model, inputs)
    first_stage_cache = model.ext[:first_stage_state]::StateCache
    create_epigraph_variables!(model, policy.policy_training_options)
    add_all_cuts!(model, policy.pool[1], policy.policy_training_options)

    store_retry_data(model, simulation_options)
    optimize_with_retry(model)
    treat_termination_status(model, simulation_options)

    for s in 1:scenarios
        future_cost = get_future_cost(model, policy.policy_training_options)
        simulation_total_cost += (JuMP.objective_value(model) - future_cost) / scenarios
        save_benders_results!(results, model, 1, s, scenarios)
    end

    # second stage
    if simulation_options.verbose
        @info("Simulating second stage...")
    end
    state = if simulation_options.state_handling == SimulationStateHandling.StatesRecalculatedInSimulation
        get_state(model)
    elseif simulation_options.state_handling == SimulationStateHandling.StatesFixedInPolicyResult
        policy.states
    else
        error("State handling not implemented.")
    end

    stage = 2
    rebuild_per_scenario =
        policy.policy_training_options.rebuild_second_stage_per_scenario
    if !rebuild_per_scenario
        state_variables_model = state_variables_builder(inputs, stage)
        model = second_stage_builder(state_variables_model, inputs)
    end

    for s in 1:scenarios
        if rebuild_per_scenario
            model = second_stage_builder(state_variables_builder(inputs, stage), inputs, s)
        end
        # Shared states plus the block of scenario s (whole vector otherwise).
        set_state(model, scenario_state(first_stage_cache, state, s))
        second_stage_modifier(model, inputs, s)

        store_retry_data(model, simulation_options; second_stage = true)
        optimize_with_retry(model)
        treat_termination_status(model, simulation_options, 2, s)

        future_cost = get_future_cost(model, policy.policy_training_options)
        simulation_total_cost += (JuMP.objective_value(model) - future_cost) / scenarios

        save_benders_results!(results, model, 2, s, scenarios)
    end

    results["objective", 0] = simulation_total_cost

    return results
end
