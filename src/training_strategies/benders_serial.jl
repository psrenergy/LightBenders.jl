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
    pool = initialize_cut_pool(policy_training_options)
    iteration_pool = initialize_cut_pool(policy_training_options)
    state = Float64[] # State variables are only stored for the first stage that does not vary per scenario
    state_cache = StateCache()

    # first stage model
    stage = 1
    state_variables_model = state_variables_builder(inputs, stage)
    first_stage_model = first_stage_builder(state_variables_model, inputs)
    create_epigraph_variables!(first_stage_model, policy_training_options)

    # second stage model
    stage = 2
    state_variables_model = state_variables_builder(inputs, stage)
    if policy_training_options.rebuild_second_stage_per_scenario
        # The second stage model is rebuilt for every scenario inside the iteration
        # loop (for problems whose structure changes with the scenario). Build the
        # scenario 1 model here only to validate that the states match.
        second_stage_model = second_stage_builder(state_variables_model, inputs, 1)
    else
        second_stage_model = second_stage_builder(state_variables_model, inputs)
    end

    check_state_match(
        first_stage_model.ext[:first_stage_state],
        second_stage_model.ext[:second_stage_state],
    )

    model_has_integrality = has_integrality(first_stage_model)
    undo_relax = relax_integrality(first_stage_model)
    relaxed = true

    best_UB_state = state

    while true
        start_iteration!(progress)
        # first stage
        if progress.current_iteration > policy_training_options.mip_options.run_mip_after_iteration && relaxed
            undo_relax()
            relaxed = false
            # Upper bounds collected with a relaxed first stage are costs of
            # fractional decisions, not valid bounds for the integer problem.
            progress.best_UB = Inf
        end
        t = 1
        add_all_cuts!(first_stage_model, iteration_pool[t], policy_training_options)
        store_retry_data(first_stage_model, policy_training_options)
        integer_active = model_has_integrality && !relaxed
        if integer_active && policy_training_options.mip_options.dynamic_gap
            gap = master_gap_target(progress, policy_training_options.mip_options)
            set_master_gap!(first_stage_model, gap)
            policy_training_options.verbose && @info "Master MIP relative gap tolerance for this iteration" gap
        end
        optimize_with_retry(first_stage_model)
        treat_termination_status(first_stage_model, policy_training_options, t, progress.current_iteration)
        state = get_state(first_stage_model; snap_tolerance = policy_training_options.state_snap_tolerance)
        future_cost = get_future_cost(first_stage_model, policy_training_options)
        progress.LB[progress.current_iteration] +=
            master_lower_bound_value(first_stage_model, policy_training_options.mip_options, integer_active)
        first_stage_cost = JuMP.objective_value(first_stage_model) - future_cost
        if policy_training_options.regularization isa LevelSetRegularization &&
           isfinite(progress.best_UB) &&
           (relaxed || !has_integrality(first_stage_model))
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
        # second stage
        t = 2
        local_pools = LocalCutPool()
        first_stage_cache = first_stage_model.ext[:first_stage_state]::StateCache
        for s in 1:policy_training_options.num_scenarios
            if policy_training_options.rebuild_second_stage_per_scenario
                second_stage_model =
                    second_stage_builder(state_variables_builder(inputs, t), inputs, s)
            end
            # Shared states plus the block of scenario s (the whole vector when
            # no state is scenario specific).
            state_s = scenario_state(first_stage_cache, state, s)
            set_state(second_stage_model, state_s)
            second_stage_modifier(second_stage_model, inputs, s)
            store_retry_data(second_stage_model, policy_training_options; second_stage = true)
            optimize_with_retry(second_stage_model)
            treat_termination_status(second_stage_model, policy_training_options, t, s, progress.current_iteration)
            coefs, rhs, obj = get_cut(second_stage_model, state_s)
            # Store the opening cut in a temporary cut pool
            store_cut!(local_pools, expand_cut_coefficients(first_stage_cache, coefs, s), state, rhs, obj)
        end
        progress.UB[progress.current_iteration] += second_stage_upper_bound_contribution(
            policy_training_options, local_pools.obj,
        )
        improved = progress.UB[progress.current_iteration] < progress.best_UB
        if improved
            progress.best_UB = progress.UB[progress.current_iteration]
            best_UB_state = copy(state)
        end
        if policy_training_options.checkpoint_callback !== nothing
            policy_training_options.checkpoint_callback(state, best_UB_state, first_stage_model, progress, improved)
        end
        if !(policy_training_options.regularization isa NoRegularization)
            # With regularization the trial states are deliberately sub-optimal for
            # the current cut pool, so the per-iteration upper bound is not
            # monotone. Report the best upper bound found so far (the paper's U^k).
            progress.UB[progress.current_iteration] = progress.best_UB
        end
        progress.time_iteration[progress.current_iteration] = time() - progress.start_time
        # Store the (stage, scenario) cut(s) in a persitent pool.
        # Cuts here can be following the single cut strategy or 
        # the multi cut strategy
        store_cut!(pool, local_pools, state, policy_training_options, t)
        store_cut!(iteration_pool, local_pools, state, policy_training_options, t)

        # check convergence
        report_current_bounds(progress)
        convergence_result =
            convergence_test(progress, policy_training_options.stopping_rule)
        if has_converged(convergence_result) || progress.LB[progress.current_iteration] > progress.UB[progress.current_iteration]
            if relaxed && model_has_integrality && !is_hard_stop(convergence_result)
                # The LP-relaxed phase converged: restore integrality and keep
                # iterating instead of returning a fractional first stage.
                @info("Relaxed first stage converged at iteration $(progress.current_iteration); restoring integrality and continuing")
                undo_relax()
                relaxed = false
                progress.best_UB = Inf
            else
                if relaxed && model_has_integrality
                    @warn("Training stopped while the first stage was still LP-relaxed; the returned state may be fractional")
                end
                finish_training!(progress, convergence_result)
                break
            end
        end
    end
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

function validate_benders_training_options(policy_training_options::PolicyTrainingOptions)
    num_errors = 0

    probs = policy_training_options.scenario_probabilities
    if !isempty(probs)
        if length(probs) != policy_training_options.num_scenarios
            @error(
                "scenario_probabilities has $(length(probs)) entries but " *
                "num_scenarios is $(policy_training_options.num_scenarios)"
            )
            num_errors += 1
        end
        if any(p -> p < 0, probs)
            @error("scenario_probabilities must be non-negative")
            num_errors += 1
        end
        if !isapprox(sum(probs), 1.0; atol = 1e-8)
            @error("scenario_probabilities must sum to 1, got $(sum(probs))")
            num_errors += 1
        end
        if policy_training_options.risk_measure isa CVaR &&
           any(p -> !isapprox(p, 1.0 / policy_training_options.num_scenarios; atol = 1e-9), probs)
            # The CVaR reformulation builds its z/delta terms assuming
            # equiprobable scenarios (the 1/((1-alpha)*L) coefficients in
            # multi_cut.jl and the deterministic equivalent). Supporting
            # non-uniform probabilities there needs the formulation reworked,
            # not just the coefficients swapped, so refuse rather than return a
            # quietly wrong answer.
            @error("CVaR with non-uniform scenario_probabilities is not supported")
            num_errors += 1
        end
    end

    if num_errors > 0
        error("Validation of policy training options failed.")
    end
    return nothing
end
