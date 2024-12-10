Base.@kwdef mutable struct Policy
    progress::AbstractProgressLog
    pool::Vector{AbstractCutPool}
    states::Vector{Float64}
    policy_training_options::PolicyTrainingOptions
end

function lower_bound(policy::Policy)
    return last_lower_bound(policy.progress)
end

function upper_bound(policy::Policy)
    return last_upper_bound(policy.progress)
end

function second_stage_upper_bound_contribution(policy_training_options::PolicyTrainingOptions, objectives::Vector{Float64})
    num_scenarios = policy_training_options.num_scenarios
    expected_value = sum(objectives[s] / num_scenarios for s in 1:num_scenarios)
    if policy_training_options.risk_measure isa RiskNeutral
        return expected_value
    elseif policy_training_options.risk_measure isa CVaR
        alpha = policy_training_options.risk_measure.alpha
        lambda = policy_training_options.risk_measure.lambda
        weights = build_cvar_weights(objectives, alpha, lambda)
        return dot(weights, objectives)
    end
end
