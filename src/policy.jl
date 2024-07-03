Base.@kwdef mutable struct Policy
    progress::AbstractProgressLog
    pool::Vector{AbstractCutPool}
    states::Matrix{Vector{Float64}}
    policy_training_options::PolicyTrainingOptions
end

function lower_bound(policy::Policy)
    return last_lower_bound(policy.progress)
end

function upper_bound(policy::Policy)
    return last_upper_bound(policy.progress)
end
