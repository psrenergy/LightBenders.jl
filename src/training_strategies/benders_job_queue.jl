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
        # second stage model
        stage = 2
        state_variables_model = state_variables_builder(inputs, stage)
        second_stage_model = second_stage_builder(state_variables_model, inputs)
        workers_loop(
            second_stage_model,
            second_stage_modifier,
            inputs,
            policy_training_options,
        )
        JQM.mpi_barrier()
        return nothing
    end

    controller = JQM.Controller(JQM.num_workers())
    progress = BendersTrainingIterationsLog(policy_training_options)
    iteration_pool = initialize_cut_pool(policy_training_options)
    state = Float64[] # State variables are only stored for the first stage that does not vary per scenario
    state_cache = StateCache()

    # first stage model
    stage = 1
    state_variables_model = state_variables_builder(inputs, stage)
    first_stage_model = first_stage_builder(state_variables_model, inputs)
    create_epigraph_variables!(first_stage_model, iteration_pool[1], policy_training_options)
    undo_relax = relax_integrality(first_stage_model)
    relaxed = true


    # second stage model (here in the controller, only used for checking if the states match)
    stage = 2
    second_stage_state_variables_model = state_variables_builder(inputs, stage)

    check_state_match(
        first_stage_model.ext[:first_stage_state],
        second_stage_state_variables_model.ext[:second_stage_state],
    )

    while true
        start_iteration!(progress)
        t = 1
        if progress.current_iteration > policy_training_options.mip_options.run_mip_after_iteration && relaxed
            undo_relax()
            relaxed = false
        end

        initialize_cuts!(first_stage_model, iteration_pool[t], policy_training_options)
        optimize_first_stage(first_stage_model, iteration_pool[t], policy_training_options, progress)
        state = get_state(first_stage_model)
        future_cost = get_future_cost(first_stage_model, policy_training_options)
        progress.LB[progress.current_iteration] += JuMP.objective_value(first_stage_model)
        progress.UB[progress.current_iteration] += JuMP.objective_value(first_stage_model) - future_cost

        t = 2
        local_pools = LocalCutPool(policy_training_options)
        for s in 1:policy_training_options.num_scenarios
            message = SecondStageMessage(progress.current_iteration, s, state)
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
                        store_cut!(
                            local_pools,
                            message.coefs,
                            state,
                            message.rhs,
                            message.obj,
                        )
                    else
                        error("Unexpected message type received from worker")
                    end
                end
            end
        end
        # Store the (stage, scenario) cut(s) in a persitent pool.
        # Cuts here can be following the single cut strategy or 
        # the multi cut strategy
        store_cut!(iteration_pool, local_pools, state, policy_training_options, t)
        reset_cuts!(first_stage_model, iteration_pool[1])
        progress.UB[progress.current_iteration] += second_stage_upper_bound_contribution(
            policy_training_options, local_pools.obj,
        )
        report_current_bounds(progress)
        convergence_result =
            convergence_test(progress, policy_training_options.stopping_rule)
        if has_converged(convergence_result)
            finish_training!(progress, convergence_result)
            JQM.send_termination_message()
            break
        end
    end
    JQM.mpi_barrier()

    return Policy(
        progress = progress,
        pool = iteration_pool,
        states = state,
        policy_training_options = policy_training_options,
    )
end

function workers_loop(
    second_stage_model::JuMP.Model,
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

        answer = worker_second_stage(
            second_stage_model,
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
