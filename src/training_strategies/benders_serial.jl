function serial_benders_train(;
    model_builder::Function,
    model_modifier::Function,
    inputs=nothing,
    policy_training_options::PolicyTrainingOptions
)
    validate_benders_training_options(policy_training_options)
    progress = BendersTrainingIterationsLog(policy_training_options)
    pool = initialize_cut_pool(policy_training_options)
    state = Float64[] # State variables are only stored for the first stage that does not vary per scenario
    state_cache = StateCache()
    while true
        start_iteration!(progress)
        for t in 1:2
            model = model_builder(inputs, t)
            # First stage does not vary per scenario
            if t == 1
                add_all_cuts!(model, pool[t], policy_training_options)
                JuMP.optimize!(model)
                treat_termination_status(model, t, 0)
                state = get_state(model)
                future_cost = get_future_cost(model, policy_training_options)
                progress.LB[progress.current_iteration] += JuMP.objective_value(model)
                progress.UB[progress.current_iteration] += JuMP.objective_value(model) - future_cost
            else
                pool_backwards = CutPoolBackwardPhase()
                for s in 1:policy_training_options.num_scenarios
                    # assume builder initializes the states
                    set_state(model, state) # State variables only exist to pass information from the first stage.
                    model_modifier(model, inputs, t, s)
                    JuMP.optimize!(model)
                    treat_termination_status(model, t, s)
                    coefs, rhs, obj = get_cut(model, state)
                    # Store the opening cut in a temporary cut pool
                    store_cut!(pool_backwards, coefs, state, rhs, obj)
                    future_cost = get_future_cost(model, policy_training_options)
                    progress.UB[progress.current_iteration] += (JuMP.objective_value(model) - future_cost) / policy_training_options.num_scenarios
                end
                # Store the (stage, scenario) cut(s) in a persitent pool.
                # Cuts here can be following the single cut strategy or 
                # the multi cut strategy
                store_cut!(pool, pool_backwards, state, policy_training_options, t)
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