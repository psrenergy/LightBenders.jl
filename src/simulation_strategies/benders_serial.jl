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
    @info("Simulating first stage...")

    state_variables_model = state_variables_builder(inputs)
    model = first_stage_builder(state_variables_model, inputs)
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
    @info("Simulating second stage...")
    state = if simulation_options.state_handling == SimulationStateHandling.StatesRecalculatedInSimulation
        get_state(model)
    elseif simulation_options.state_handling == SimulationStateHandling.StatesFixedInPolicyResult
        policy.states
    else
        error("State handling not implemented.")
    end

    state_variables_model = state_variables_builder(inputs)
    model = second_stage_builder(state_variables_model, inputs)
    set_state(model, state)

    for s in 1:scenarios
        second_stage_modifier(model, inputs, s)

        store_retry_data(model, simulation_options)
        optimize_with_retry(model)
        treat_termination_status(model, simulation_options, 2, s)

        future_cost = get_future_cost(model, policy.policy_training_options)
        simulation_total_cost += (JuMP.objective_value(model) - future_cost) / scenarios

        save_benders_results!(results, model, 2, s, scenarios)
    end

    results["objective", 0] = simulation_total_cost

    return results
end
