abstract type AbstractTrainingImplementation end

struct BendersSerialTraining <: AbstractTrainingImplementation end

"""
"""
Base.@kwdef mutable struct PolicyTrainingOptions
    num_scenarios::Int
    lower_bound::Real = 0.0
    discount_rate::Real = 0.0
    implementation_strategy::AbstractTrainingImplementation = BendersSerialTraining()
    cut_strategy::CutStrategy.T = CutStrategy.SingleCut
    risk_measure::AbstractRiskMeasure = RiskNeutral()
    stopping_rule::AbstractStoppingRule = IterationLimit(5)
    logs_dir::String = ""
    write_lp::Bool = false
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
    policy_training_options::PolicyTrainingOptions
)
    if policy_training_options.implementation_strategy isa BendersSerialTraining
        return serial_benders_train(;
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
