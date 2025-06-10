function serial_benders_train(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    policy_training_options::PolicyTrainingOptions,
)
    validate_benders_training_options(policy_training_options)
    progress = BendersTrainingIterationsLog(policy_training_options)
    iteration_pool = initialize_cut_pool(policy_training_options)
    state = Float64[] # State variables are only stored for the first stage that does not vary per scenario
    state_cache = StateCache()

    # first stage model
    stage = 1
    state_variables_model = state_variables_builder(inputs, stage)
    first_stage_model = first_stage_builder(state_variables_model, inputs)
    create_epigraph_variables!(first_stage_model, iteration_pool[1], policy_training_options)

    # second stage model
    stage = 2
    state_variables_model = state_variables_builder(inputs, stage)
    second_stage_model = second_stage_builder(state_variables_model, inputs)

    check_state_match(
        first_stage_model.ext[:first_stage_state],
        second_stage_model.ext[:second_stage_state],
    )

    while true
        start_iteration!(progress)
        # first stage
        t = 1
        initialize_cuts!(first_stage_model, iteration_pool[t], policy_training_options)
        optimize_first_stage(first_stage_model, iteration_pool[t], policy_training_options, progress)
        state = get_state(first_stage_model)
        future_cost = get_future_cost(first_stage_model, policy_training_options)
        progress.LB[progress.current_iteration] += JuMP.objective_value(first_stage_model)
        progress.UB[progress.current_iteration] += JuMP.objective_value(first_stage_model) - future_cost

        # second stage
        t = 2
        local_pools = LocalCutPool(policy_training_options)
        for s in 1:policy_training_options.num_scenarios
            set_state(second_stage_model, state)
            second_stage_modifier(second_stage_model, inputs, s)
            store_retry_data(second_stage_model, policy_training_options)
            optimize_with_retry(second_stage_model)
            treat_termination_status(second_stage_model, policy_training_options, t, s, progress.current_iteration)
            coefs, rhs, obj = get_cut(second_stage_model, state)
            # Store the opening cut in a temporary cut pool
            store_cut!(local_pools, coefs, state, rhs, obj)
        end
        progress.UB[progress.current_iteration] += second_stage_upper_bound_contribution(
            policy_training_options, local_pools.obj,
        )
        # Store the (stage, scenario) cut(s) in a persitent pool.
        # Cuts here can be following the single cut strategy or 
        # the multi cut strategy

        store_cut!(iteration_pool, local_pools, state, policy_training_options, t)
        # Reset the cut pool for the next iteration
        reset_cuts!(first_stage_model, iteration_pool[1])

        # check convergence
        report_current_bounds(progress)
        convergence_result =
            convergence_test(progress, policy_training_options.stopping_rule)
        if has_converged(convergence_result)
            finish_training!(progress, convergence_result)
            break
        end
    end
    return Policy(
        progress = progress,
        pool = iteration_pool,
        states = state,
        policy_training_options = policy_training_options,
    )
end

function validate_benders_training_options(policy_training_options::PolicyTrainingOptions)
    num_errors = 0

    if num_errors > 0
        error("Validation of policy training options failed.")
    end
    return nothing
end
