function serial_benders_train(;
    model_builder::Function,
    model_modifier::Function,
    inputs=nothing,
    policy_training_options::PolicyTrainingOptions
)
    validate_benders_training_options(policy_training_options)
    progress = BendersTrainingIterationsLog(policy_training_options)
    pool = initialize_cut_pool(policy_training_options)
    state = [Float64[] for _ in 1:(policy_training_options.num_stages+1), _ in 1:1] # State variables are only stored for the first stage that does not vary per scenario
    state_cache_in = [StateCache() for t in 1:policy_training_options.num_stages]
    state_cache_out = [StateCache() for t in 1:(policy_training_options.num_stages+1)]
    while true
        start_iteration!(progress)
        # forward pass
        for t in 1:2
            model = model_builder(inputs, t)
            # for state validation
            store_state_cache(state_cache_in, state_cache_out, model, t)
            # First stage does not vary per scenario
            if t == 1
                add_all_cuts!(model, pool[t], policy_training_options)
                model_modifier(model, inputs, t, 0, 0)
                JuMP.optimize!(model)
                treat_termination_status(model, t, 0)
                state[t+1, 1] = get_state(model)
                future_cost = get_future_cost(model, policy_training_options)
                progress.LB[progress.current_iteration] += JuMP.objective_value(model)
                progress.UB[progress.current_iteration] += JuMP.objective_value(model) - future_cost
            else
                for s in 1:policy_training_options.num_scenarios
                    # assume builder initializes the states
                    set_state(model, state[t, 1]) # State variables only exist to pass information from the first stage.
                    model_modifier(model, inputs, t, s, 0)
                    JuMP.optimize!(model)
                    treat_termination_status(model, t, s)
                    future_cost = get_future_cost(model, policy_training_options)
                    progress.UB[progress.current_iteration] += (JuMP.objective_value(model) - future_cost) / policy_training_options.num_scenarios
                end
            end
        end
        report_current_bounds(progress)
        convergence_result =
            convergence_test(progress, policy_training_options.stopping_rule)
        if has_converged(convergence_result)
            println(message(convergence_result))
            finish_training!(progress)
            break
        end
        # backward pass
        # In the benders implementation it only runs on the second stage to generate cuts
        t = 2
        model = model_builder(inputs, t)
        set_state(model, state[t, 1])
        state[t, 1]
        pool_backwards = CutPoolBackwardPhase()
        for s in 1:policy_training_options.num_scenarios
            model_modifier(model, inputs, t, s, 0)
            JuMP.optimize!(model)
            treat_termination_status(model, t, s, 0)
            coefs, rhs, obj = get_cut(model, state[t, 1])
            # Store the opening cut in a temporary cut pool
            store_cut!(pool_backwards, coefs, state[t, 1], rhs, obj)
        end
        # Store the (stage, scenario) cut(s) in a persitent pool.
        # Cuts here can be following the single cut strategy or 
        # the multi cut strategy
        store_cut!(pool, pool_backwards, state[t, 1], policy_training_options, t)
    end
    return Policy(
        progress=progress,
        pool=pool,
        states=state,
        policy_training_options=policy_training_options
    )
end

function validate_benders_training_options(policy_training_options::PolicyTrainingOptions)
    num_errors = 0

    if policy_training_options.num_stages != 2
        @error "Benders implementation only accepts cases with two stages."
        num_errors += 1
    end
    if policy_training_options.num_openings != policy_training_options.num_scenarios
        @error "In Benders implementations the number of scenarios is equal to the number of openings."
        num_errors += 1
    end
    if num_errors > 0
        error("Validation of policy training options failed.")
    end
    return nothing
end