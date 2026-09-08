abstract type AbstractTrainingImplementation end

struct SerialTraining <: AbstractTrainingImplementation end

struct JobQueueTraining <: AbstractTrainingImplementation end

"""
"""
Base.@kwdef mutable struct PolicyTrainingOptions
    num_scenarios::Int
    # Probability of each scenario. Empty means equiprobable (1/num_scenarios),
    # which is what every earlier version assumed unconditionally.
    scenario_probabilities::Vector{Float64} = Float64[]
    lower_bound::Real = 0.0
    discount_rate::Real = 0.0
    verbose::Bool = true
    implementation_strategy::AbstractTrainingImplementation = SerialTraining()
    cut_strategy::CutStrategy.T = CutStrategy.SingleCut
    risk_measure::AbstractRiskMeasure = RiskNeutral()
    regularization::AbstractRegularization = NoRegularization()
    rebuild_second_stage_per_scenario::Bool = false
    # When set, one row per iteration (bounds, gap, time) is appended and
    # flushed to this file during training, regardless of `verbose`.
    progress_log_file::String = ""
    stopping_rule::Vector{AbstractStoppingRule} = [IterationLimit(5)]
    mip_options::MIPOptions = MIPOptions()
    debugging_options::DebuggingOptions = DebuggingOptions()
    retry_optimize::RetryOptimizeOptions = RetryOptimizeOptions()
    # see STATE_SNAP_TOLERANCE
    state_snap_tolerance::Float64 = STATE_SNAP_TOLERANCE
    # Called at the end of every iteration as
    # `checkpoint_callback(state, best_state, first_stage_model, progress, improved)`, where
    # `state` is this iteration's trial first-stage decision, `best_state` the one with the
    # best upper bound so far and `improved` whether it changed this iteration. Lets the
    # caller persist decisions so a long run can be stopped without losing them and its
    # evolution can be inspected.
    checkpoint_callback::Union{Function, Nothing} = nothing
end

"""
    train(;
        state_variables_builder::Function,
        first_stage_builder::Function,
        second_stage_builder::Function,
        second_stage_modifier::Function,
        inputs,
        options::PolicyTrainingOptions
    )

Train a policy using the Benders algorithm. There are various ways of performing this training,
which are controlled by the options argument. The `state_variables_builder`, `first_stage_builder`, `second_stage_builder` and `second_stage_modifier`
functions are used to construct and modify the models that are used in the training process.
The inputs argument contains the data that is used to build the models.
"""
function train(;
    state_variables_builder::Function,
    first_stage_builder::Function,
    second_stage_builder::Function,
    second_stage_modifier::Function,
    inputs = nothing,
    policy_training_options::PolicyTrainingOptions,
)
    if policy_training_options.implementation_strategy isa SerialTraining
        return serial_benders_train(;
            state_variables_builder,
            first_stage_builder,
            second_stage_builder,
            second_stage_modifier,
            inputs,
            policy_training_options,
        )
    elseif policy_training_options.implementation_strategy isa JobQueueTraining
        return job_queue_benders_train(;
            state_variables_builder,
            first_stage_builder,
            second_stage_builder,
            second_stage_modifier,
            inputs,
            policy_training_options,
        )
    else
        error("ImplementationStrategy not implemented.")
    end
end
