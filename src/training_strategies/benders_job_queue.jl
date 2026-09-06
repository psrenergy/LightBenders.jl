mutable struct SecondStageMessage
    iteration::Int
    scenario::Int
    state::Any
end

mutable struct SecondStageAnswer
    coefs::Any
    rhs::Any
    obj::Any
    scenario::Int
end

function job_queue_benders_train(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    policy_training_options::PolicyTrainingOptions,
)
    JQM.mpi_init()
    JQM.mpi_barrier()

    validate_benders_training_options(policy_training_options)

    if JQM.is_worker_process()
        # second stage model. With rebuild_second_stage_per_scenario the model is
        # built per received job (the structure changes with the scenario), so
        # nothing is prebuilt here.
        stage = 2
        second_stage_model = if policy_training_options.rebuild_second_stage_per_scenario
            nothing
        else
            state_variables_model = state_variables_builder(inputs, stage)
            second_stage_builder(state_variables_model, inputs)
        end
        workers_loop(
            second_stage_model,
            state_variables_builder,
            second_stage_builder,
            second_stage_modifier,
            inputs,
            policy_training_options,
        )
        JQM.mpi_barrier()
        return nothing
    end

    controller = JQM.Controller(JQM.num_workers())
    progress = BendersTrainingIterationsLog(policy_training_options)
    pool = initialize_cut_pool(policy_training_options)
    iteration_pool = initialize_cut_pool(policy_training_options)
    state = Float64[] # State variables are only stored for the first stage that does not vary per scenario
    state_cache = StateCache()

    # first stage model
    stage = 1
    state_variables_model = state_variables_builder(inputs, stage)
    first_stage_model = first_stage_builder(state_variables_model, inputs)
    create_epigraph_variables!(first_stage_model, policy_training_options)
    if policy_training_options.mip_options.run_mip_after_iteration > 0
        undo_relax = relax_integrality(first_stage_model)
        relaxed = true
    end

    # second stage model (here in the controller, only used for checking if the states match)
    stage = 2
    second_stage_state_variables_model = state_variables_builder(inputs, stage)
    if policy_training_options.rebuild_second_stage_per_scenario
        # State registration happens inside the scenario builder — build the
        # scenario 1 model once to validate the states, then discard it.
        second_stage_state_variables_model =
            second_stage_builder(second_stage_state_variables_model, inputs, 1)
    end

    check_state_match(
        first_stage_model.ext[:first_stage_state],
        second_stage_state_variables_model.ext[:second_stage_state],
    )

    best_UB_state = state

    while true
        start_iteration!(progress)
        t = 1
        if policy_training_options.mip_options.run_mip_after_iteration > 0
            if progress.current_iteration > policy_training_options.mip_options.run_mip_after_iteration && relaxed
                undo_relax()
                relaxed = false
            end
        end
        add_all_cuts!(first_stage_model, iteration_pool[t], policy_training_options)
        store_retry_data(first_stage_model, policy_training_options)
        optimize_with_retry(first_stage_model)
        treat_termination_status(first_stage_model, policy_training_options, t, progress.current_iteration)
        state = get_state(first_stage_model)
        future_cost = get_future_cost(first_stage_model, policy_training_options)
        progress.LB[progress.current_iteration] += JuMP.objective_value(first_stage_model)
        first_stage_cost = JuMP.objective_value(first_stage_model) - future_cost
        if policy_training_options.regularization isa LevelSetRegularization &&
           isfinite(progress.best_UB) &&
           ((policy_training_options.mip_options.run_mip_after_iteration > 0 && relaxed) ||
            !has_integrality(first_stage_model))
            level_set_solution = solve_level_set_problem!(
                first_stage_model,
                policy_training_options,
                progress.LB[progress.current_iteration],
                progress.best_UB,
            )
            if level_set_solution !== nothing
                state, first_stage_cost = level_set_solution
            end
        end
        progress.UB[progress.current_iteration] += first_stage_cost
        iteration_pool = initialize_cut_pool(policy_training_options)

        t = 2
        # Pre-allocate LocalCutPool to store cuts in correct scenario order
        local_pools = LocalCutPool(policy_training_options.num_scenarios)
        first_stage_cache = first_stage_model.ext[:first_stage_state]::StateCache
        for s in 1:policy_training_options.num_scenarios
            # Workers receive only the states their scenario needs.
            message = SecondStageMessage(
                progress.current_iteration, s, scenario_state(first_stage_cache, state, s),
            )
            JQM.add_job_to_queue!(controller, message)
        end
        while JQM.any_jobs_left(controller)
            if !JQM.is_job_queue_empty(controller)
                JQM.send_jobs_to_any_available_workers(controller)
            end
            if JQM.any_pending_jobs(controller)
                job_answer = JQM.check_for_job_answers(controller)
                if !isnothing(job_answer)
                    message = JQM.get_message(job_answer)
                    if message isa SecondStageAnswer
                        # Store cut at correct scenario index to preserve ordering
                        store_cut!(
                            local_pools,
                            expand_cut_coefficients(first_stage_cache, message.coefs, message.scenario),
                            state,
                            message.rhs,
                            message.obj,
                            message.scenario,
                        )
                    else
                        error("Unexpected message type received from worker")
                    end
                end
            end
        end
        # Validate that all scenarios completed successfully
        validate_all_scenarios_processed(local_pools, policy_training_options.num_scenarios)
        # Store the (stage, scenario) cut(s) in a persistent pool.
        # Cuts here can be following the single cut strategy or
        # the multi cut strategy
        store_cut!(pool, local_pools, state, policy_training_options, t)
        store_cut!(iteration_pool, local_pools, state, policy_training_options, t)
        progress.UB[progress.current_iteration] += second_stage_upper_bound_contribution(
            policy_training_options, local_pools.obj,
        )
        if progress.UB[progress.current_iteration] < progress.best_UB
            progress.best_UB = progress.UB[progress.current_iteration]
            best_UB_state = copy(state)
        end
        if !(policy_training_options.regularization isa NoRegularization)
            # With regularization the trial states are deliberately sub-optimal for
            # the current cut pool, so the per-iteration upper bound is not
            # monotone. Report the best upper bound found so far (the paper's U^k).
            progress.UB[progress.current_iteration] = progress.best_UB
        end
        progress.time_iteration[progress.current_iteration] = time() - progress.start_time
        report_current_bounds(progress)
        convergence_result =
            convergence_test(progress, policy_training_options.stopping_rule)
        if has_converged(convergence_result) || progress.LB[progress.current_iteration] > progress.UB[progress.current_iteration]
            finish_training!(progress, convergence_result)
            JQM.send_termination_message()
            break
        end
    end
    JQM.mpi_barrier()

    # With regularization the last state is a level-set trial point, so the plan
    # that achieved the best upper bound is the one to return.
    final_state =
        policy_training_options.regularization isa NoRegularization ? state : best_UB_state
    return Policy(
        progress = progress,
        pool = pool,
        states = final_state,
        policy_training_options = policy_training_options,
    )
end

function workers_loop(
    second_stage_model::Union{JuMP.Model, Nothing},
    state_variables_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs,
    policy_training_options::PolicyTrainingOptions,
)
    worker = JQM.Worker()
    while true
        # Check if any job was sent
        job = JQM.receive_job(worker)
        message = JQM.get_message(job)
        if message == JQM.TerminationMessage()
            break
        end

        model = if policy_training_options.rebuild_second_stage_per_scenario
            second_stage_builder(state_variables_builder(inputs, 2), inputs, message.scenario)
        else
            second_stage_model
        end
        answer = worker_second_stage(
            model,
            second_stage_modifier,
            inputs,
            policy_training_options,
            message,
        )
        JQM.send_job_answer_to_controller(worker, answer)
    end
    return nothing
end

function worker_second_stage(
    second_stage_model::JuMP.Model,
    second_stage_modifier,
    inputs,
    policy_training_options,
    message,
)
    t = 2
    scenario = message.scenario
    iteration = message.iteration
    state = message.state
    # We could only build the model once and modify it for each scenario

    set_state(second_stage_model, state)
    second_stage_modifier(second_stage_model, inputs, scenario)
    store_retry_data(second_stage_model, policy_training_options)
    optimize_with_retry(second_stage_model)
    treat_termination_status(second_stage_model, policy_training_options, t, scenario, iteration)
    coefs, rhs, obj = get_cut(second_stage_model, state)

    return SecondStageAnswer(
        coefs,
        rhs,
        obj,
        scenario,
    )
end
