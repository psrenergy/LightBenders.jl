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

    stages = 2
    scenarios = simulation_options.num_scenarios

    simulation_total_cost = 0.0

    state = Float64[]
    for t in 1:stages
        model = model_builder(inputs, t)::JuMP.Model
        if t != stages
            add_all_cuts!(model, policy.pool[t], policy.policy_training_options)
        end
        for s in 1:scenarios
            if t != 1
                set_state(model, state)
            end
            model_modifier(model, inputs, t, s)::Nothing
            JuMP.optimize!(model)
            treat_termination_status(model, t, s)
            future_cost = get_future_cost(model, policy.policy_training_options)
            simulation_total_cost += (JuMP.objective_value(model) - future_cost) / scenarios
            if results_recorder !== nothing
                results_recorder(model, inputs, t, s, simulation_options.outputs_path)::Nothing
            end
            if simulation_options.state_handling == SimulationStateHandling.StatesRecalculatedInSimulation
                state = get_state(model)
            elseif simulation_options.state_handling == SimulationStateHandling.StatesFixedInPolicyResult
                state = policy.states
            else
                error("State handling not implemented.")
            end
        end
    end
    if results_recorder !== nothing && simulation_options.gather_outputs
        results_gatherer(stages, collect(1:scenarios), simulation_options.outputs_path)
    end
    return simulation_total_cost
end