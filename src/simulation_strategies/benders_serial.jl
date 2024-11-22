function serial_benders_simulate(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs=nothing,
    policy::Policy,
    simulation_options::SimulationOptions,
)
    stages = 2
    scenarios = simulation_options.num_scenarios

    simulation_total_cost = 0.0

    state = Float64[]

    results = Dict{Tuple{String, Int}, Any}() # (variable_name, scenario) => value

    p = Progress(
        stages, 
        barlen=60,
        barglyphs=BarGlyphs("[=> ]"),
        color=:white,
        desc="",
    )

    for t in 1:stages
        if t == 1 # first stage
            state_variables_model = state_variables_builder(inputs)
            model = first_stage_builder(state_variables_model, inputs)
            add_all_cuts!(model, policy.pool[t, policy.policy_training_options)
        elseif t == 2 # second stage
            state_variables_model = state_variables_builder(inputs)
            model = second_stage_builder(state_variables_model, inputs)
            set_state(model, state)
        end
        for s in 1:scenarios
            if t == 2
                second_stage_modifier(model, inputs, s)
            end
            store_retry_data(model, simulation_options)
            optimize_with_retry(model)
            treat_termination_status(model, simulation_options, t, s)
            future_cost = get_future_cost(model, policy.policy_training_options)
            simulation_total_cost += (JuMP.objective_value(model) - future_cost) / scenarios
            save_benders_results!(results, model, t, s, scenarios)
            if simulation_options.state_handling == SimulationStateHandling.StatesRecalculatedInSimulation
                state = get_state(model)
            elseif simulation_options.state_handling == SimulationStateHandling.StatesFixedInPolicyResult
                state = policy.states
            else
                error("State handling not implemented.")
            end
        end

        next!(p)
    end

    finish!(p)

    results["objective", 0] = simulation_total_cost

    return results
end
