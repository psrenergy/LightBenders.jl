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
        if isnothing(policy_training_options.scenario_map)
            weights = build_cvar_weights(objectives, alpha, lambda)
            return dot(weights, objectives)
        else
            scenario_map = policy_training_options.scenario_map
            group_obj, group_counts = aggregate_by_group(objectives, scenario_map)
            weights_on_groups = build_cvar_weights(group_obj, alpha, lambda)
            subproblem_weights = [weights_on_groups[scenario_map[s]] / group_counts[scenario_map[s]] for s in eachindex(scenario_map)]
            return dot(subproblem_weights, objectives)
        end
    end
end
