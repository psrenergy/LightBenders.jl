"""
    AbstractRiskMeasure

Abstract type to hold risk measure implementations.
"""
abstract type AbstractRiskMeasure end

"""
    RiskNeutral

Options for the risk neutral risk measure, the risk measure is the expected value.
"""
struct RiskNeutral <: AbstractRiskMeasure end

"""
    CVaR

CVaR risk measure, the risk measure is a convex combination of the expected value and the CVaR.

# Fields
- `alpha::Float64`: The confidence level for the CVaR
- `lambda::Float64`: The weight of the CVaR in the convex combination

The convex combination is expressed as:
```math
\\begin{equation}
    (1 - \\lambda) \\text{Expected Value} + \\lambda \\text{CVaR}_{\\alpha}
\\end{equation}
```

Example: 
```julia
cvar_options = CVaROptions(alpha=0.95, lambda=0.5)
```
will give a risk measure that is a convex combination of the expected value and the CVaR with a confidence level of 0.95 and a weight of 0.5.
```math
\\begin{equation}
    0.5 \\text{Expected Value} + 0.5 \\text{CVaR}_{0.95}
\\end{equation}
```
Where the CVaR of 95% is approximatelly equal to the mean of the 5% worst cases.

If we choose `lambda=0.0` the risk measure will be the expected value. If we choose `lambda=1.0` the risk measure will be the CVaR.
If we choose `alpha=0.0` the CVaR will be equivalent to the expected value. If we choose `alpha=1.0` the CVaR will be the worst case scenario.
"""
mutable struct CVaR <: AbstractRiskMeasure
    alpha::Float64
    lambda::Float64
    function CVaR(;alpha::Real=0.95, lambda::Real=0.5)::CVaR
        num_errors = 0
        if alpha < 0.0 || alpha > 1.0
            @error("alpha must be between 0 and 1.")
            num_errors += 1
        end
        if lambda < 0.0 || lambda > 1.0
            @error("lambda must be between 0 and 1.")
            num_errors += 1
        end
        if num_errors > 0
            throw(ArgumentError("Invalid arguments in CVaROptions."))
        end
        return new(alpha, lambda)
    end
end

function build_cvar_weights(
    objectives::Vector{Float64}, 
    alpha::Real, 
    lambda::Real, 
)::Vector{Float64}
    N = length(objectives)
    κ = ceil(alpha * N)
    I = sortperm(objectives)
    weights = zeros(Float64, N)
    # For each scenario,
    # find its weight ordered by frequencies
    weight_under_k = (1 - lambda) / N
    weight_k = (1 - lambda) / N + lambda - (lambda * (N - κ) / ((1 - alpha) * N))
    weight_over_k = (1 - lambda) / N + lambda / ((1 - alpha) * N)
    for (i, real_i) in enumerate(I)
        if i < κ
            weights[real_i] = weight_under_k
        elseif i == κ
            weights[real_i] = weight_k
        else
            weights[real_i] = weight_over_k
        end
    end
    return weights
end