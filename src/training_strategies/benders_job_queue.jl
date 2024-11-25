mutable struct FirstStageMessage
    pool
    iteration::Int
end

mutable struct FirstStageAnswer
    state
    LB::Float64
    UB::Float64
end

mutable struct SecondStageMessage
    iteration::Int
    scenario::Int
    state
end

mutable struct SecondStageAnswer
    UB::Float64
    coefs
    rhs
    obj
end

all_jobs_done(controller) = JQM.is_job_queue_empty(controller) && !JQM.any_pending_jobs(controller)

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
        workers_loop(
            state_variables_builder,
            first_stage_builder,
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
    state = Float64[] # State variables are only stored for the first stage that does not vary per scenario
    state_cache = StateCache()

    while true
        start_iteration!(progress)
        t = 1
        message = FirstStageMessage(pool, progress.current_iteration)
        JQM.add_job_to_queue!(controller, message)
        JQM.send_jobs_to_any_available_workers(controller)
        # Wait for the answer
        while true
            job_answer = JQM.check_for_job_answers(controller)
            if !isnothing(job_answer)
                message = JQM.get_message(job_answer)
                if message isa FirstStageAnswer
                    state = message.state
                    progress.LB[progress.current_iteration] += message.LB
                    progress.UB[progress.current_iteration] += message.UB
                    break
                else
                    error("Unexpected message type received from worker")
                end
            end
        end

        t = 2
        local_pools = LocalCutPool()
        for s in 1:policy_training_options.num_scenarios
            message = SecondStageMessage(progress.current_iteration, s, state)
            JQM.add_job_to_queue!(controller, message)
        end
        while !all_jobs_done(controller)
            if !JQM.is_job_queue_empty(controller)
                JQM.send_jobs_to_any_available_workers(controller)
            end
            if JQM.any_pending_jobs(controller)
                job_answer = JQM.check_for_job_answers(controller)
                if !isnothing(job_answer)
                    message = JQM.get_message(job_answer)
                    if message isa SecondStageAnswer
                        progress.UB[progress.current_iteration] += message.UB
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
        store_cut!(pool, local_pools, state, policy_training_options, t)
        report_current_bounds(progress)
        convergence_result =
            convergence_test(progress, policy_training_options.stopping_rule)
        if has_converged(convergence_result)
            println(results_message(convergence_result))
            finish_training!(progress)
            JQM.send_termination_message()
            break
        end
    end
    JQM.mpi_barrier()
    return Policy(
        progress = progress,
        pool = pool,
        states = state,
        policy_training_options = policy_training_options,
    )
end

function workers_loop(
    state_variables_builder::Function,
    first_stage_builder::Function,
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
        if message isa FirstStageMessage
            answer = worker_first_stage(
                state_variables_builder,
                first_stage_builder,
                second_stage_builder,
                second_stage_modifier,
                inputs,
                policy_training_options,
                message,
            )
            JQM.send_job_answer_to_controller(worker, answer)
        elseif message isa SecondStageMessage
            answer = worker_second_stage(
                state_variables_builder,
                first_stage_builder,
                second_stage_builder,
                second_stage_modifier,
                inputs,
                policy_training_options,
                message,
            )
            JQM.send_job_answer_to_controller(worker, answer)
        end
    end
    return nothing
end

function worker_first_stage(
    state_variables_builder,
    first_stage_builder,
    second_stage_builder,
    second_stage_modifier,
    inputs,
    policy_training_options,
    message,
)
    pool = message.pool
    iteration = message.iteration
    t = 1
    state_variables_model = state_variables_builder(inputs)
    first_stage_model = first_stage_builder(state_variables_model, inputs)
    add_all_cuts!(first_stage_model, pool[t], policy_training_options)
    store_retry_data(first_stage_model, policy_training_options)
    optimize_with_retry(first_stage_model)
    treat_termination_status(first_stage_model, policy_training_options, t, iteration)
    state = get_state(first_stage_model)
    future_cost = get_future_cost(first_stage_model, policy_training_options)
    LB = JuMP.objective_value(first_stage_model)
    UB = JuMP.objective_value(first_stage_model) - future_cost
    return FirstStageAnswer(
        state,
        LB,
        UB,
    )
end

function worker_second_stage(
    state_variables_builder,
    first_stage_builder,
    second_stage_builder,
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
    state_variables_model = state_variables_builder(inputs)
    second_stage_model = second_stage_builder(state_variables_model, inputs)
    set_state(second_stage_model, state)
    second_stage_modifier(second_stage_model, inputs, scenario)
    store_retry_data(second_stage_model, policy_training_options)
    optimize_with_retry(second_stage_model)
    treat_termination_status(second_stage_model, policy_training_options, t, scenario, iteration)
    coefs, rhs, obj = get_cut(second_stage_model, state)
    future_cost = get_future_cost(second_stage_model, policy_training_options)
    UB = (JuMP.objective_value(second_stage_model) - future_cost) / policy_training_options.num_scenarios
    return SecondStageAnswer(
        UB,
        coefs,
        rhs,
        obj,
    )
end
