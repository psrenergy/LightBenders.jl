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
end

"""
    train(;
        model_builder::Function,
        model_modifier::Function,
        inputs,
        options::PolicyTrainingOptions
    )

Train a policy using the Benders algorithm. There are various ways of performing this training,
which are controlled by the options argument. The `model_builder` and `model_modifier` 
functions are used to construct and modify the models that are used in the training process. 
The inputs argument contains the data that is used to build the models.
"""
function train(;
    model_builder::Function,
    model_modifier::Function,
    inputs = nothing,
    policy_training_options::PolicyTrainingOptions
)
    if policy_training_options.implementation_strategy isa BendersSerialTraining
        return serial_benders_train(;model_builder, model_modifier, inputs, policy_training_options)
    else
        error("ImplementationStrategy not implemented.")
    end
end