function serial_benders_simulate(;
    model_builder::Function,
    model_modifier::Function,
    results_recorder::Union{Function,Nothing}=nothing,
    inputs=nothing,
    policy::Policy,
    simulation_options::SimulationOptions
)
    if results_recorder !== nothing
        if ispath(simulation_options.outputs_path)
            rm(simulation_options.outputs_path; recursive=true, force=true)
        end
        mkpath(simulation_options.outputs_path)
    end

    stages = simulation_options.num_stages
    scenarios = simulation_options.num_scenarios

    simulation_total_cost = 0.0

    state = [Float64[] for t in 1:(stages+1), s in 1:scenarios]
    state_cache_in = [StateCache() for t in 1:stages]
    state_cache_out = [StateCache() for t in 1:(stages+1)]
    # forward pass
    for t in 1:stages
        model = model_builder(inputs, t)::JuMP.Model
        # for state validation
        store_state_cache(state_cache_in, state_cache_out, model, t)
        if t != stages
            add_all_cuts!(model, policy.pool[t], policy.policy_training_options)
        end
        for s in 1:scenarios
            if t != 1
                set_state(model, state[t, 1])
            end
            model_modifier(model, inputs, t, s, 0)::Nothing
            JuMP.optimize!(model)
            treat_termination_status(model, t, s)
            future_cost = get_future_cost(model, policy.policy_training_options)
            simulation_total_cost += (JuMP.objective_value(model) - future_cost) / scenarios
            if results_recorder !== nothing
                results_recorder(model, inputs, t, s, simulation_options.outputs_path)::Nothing
            end
            if simulation_options.state_handling == SimulationStateHandling.StatesRecalculatedInSimulation
                state[t+1, 1] = get_state(model)
            elseif simulation_options.state_handling == SimulationStateHandling.StatesFixedInPolicyResult
                state[t+1, 1] = policy.states[t+1, 1]
            else
                error("State handling not implemented.")
            end
        end
        # state validation
        if t > 1
            check_state_match(state_cache_in, state_cache_out, t)
        end
    end
    if results_recorder !== nothing && simulation_options.gather_outputs
        results_gatherer(stages, collect(1:scenarios), simulation_options.outputs_path)
    end
    return simulation_total_cost
end