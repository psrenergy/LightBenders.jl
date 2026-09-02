function truncate_small_numbers(x::Float64)
    if isapprox(x, 0.0, atol = 1e-6)
        return 0.0
    else
        return x
    end
end

"""
    scenario_probabilities(policy_training_options) -> Vector{Float64}

Per-scenario probabilities, defaulting to uniform `1/num_scenarios` when none
were supplied. Every weighted sum over scenarios — cut aggregation, the
epigraph objective coefficients, the upper bound, the deterministic equivalent
— goes through this, so a case with non-uniform probabilities is weighted
consistently everywhere.
"""
function scenario_probabilities(policy_training_options)
    probs = policy_training_options.scenario_probabilities
    n = policy_training_options.num_scenarios
    isempty(probs) && return fill(1.0 / n, n)
    return probs
end

"""
    scenario_probability(policy_training_options, scenario::Int) -> Float64

Probability of a single scenario. See [`scenario_probabilities`](@ref).
"""
function scenario_probability(policy_training_options, scenario::Int)
    probs = policy_training_options.scenario_probabilities
    isempty(probs) && return 1.0 / policy_training_options.num_scenarios
    return probs[scenario]
end
