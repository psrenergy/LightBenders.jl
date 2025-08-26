"""
    Abastract type to hold stopping rules implementations.
"""
abstract type AbstractStoppingRule end

struct ConvergenceResult
    has_converged::Bool
    message::String
end

has_converged(result::ConvergenceResult) = result.has_converged
results_message(result::ConvergenceResult) = result.message

"""
    IterationLimit(max_iterations::Int)

Terminate the algorithm when the number of iterations reaches `max_iterations`.
"""
Base.@kwdef struct IterationLimit <: AbstractStoppingRule
    max_iterations::Int
end

function convergence_test(
    progress,
    rules::Vector{<:AbstractStoppingRule},
)
    for rule in rules
        result = convergence_test(progress, rule)
        if has_converged(result)
            return result
        end
    end
    return ConvergenceResult(false, "not converged.")
end

function convergence_test(
    progress,
    rule::IterationLimit,
)
    has_converged = progress.current_iteration >= rule.max_iterations
    return ConvergenceResult(has_converged, "converged with $rule.")
end

"""
    Gap(abstol::Number, reltol::Number)

Terminate the algorithm when the lower and upper bounds are within `abstol` or `reltol` and it has run at least .
"""
Base.@kwdef struct Gap <: AbstractStoppingRule
    abstol::Number = 1e-6
    reltol::Number = 1e-3
end

function convergence_test(
    progress,
    rule::Gap,
)
    has_converged = isapprox(last_lower_bound(progress), last_upper_bound(progress); atol = rule.abstol, rtol = rule.reltol)
    return ConvergenceResult(has_converged, "converged with gap.")
end

"""
    GapWithMinimumNumberOfIterations(abstol::Number, reltol::Number, min_iterations::Int)

Terminate the algorithm when the lower and upper bounds are within `abstol` or `reltol` and it has run at least `min_iterations`.
"""
Base.@kwdef struct GapWithMinimumNumberOfIterations <: AbstractStoppingRule
    abstol::Number = 1e-6
    reltol::Number = 1e-3
    min_iterations::Int = 10
end

function convergence_test(
    progress,
    rule::GapWithMinimumNumberOfIterations,
)
    both_converged = progress.current_iteration >= rule.min_iterations && has_converged(convergence_test(progress, Gap(rule.abstol, rule.reltol)))
    return ConvergenceResult(both_converged, "converged with gap with minimum number of iterations")
end

"""
    TimeLimit(seconds::Int)

Terminate the algorithm when the elapsed time exceeds `seconds`.
"""

Base.@kwdef struct TimeLimit <: AbstractStoppingRule
    seconds::Int = 3600  # Default to 1 hour
end

function convergence_test(
    progress,
    rule::TimeLimit,
)
    has_converged = (time() - progress.start_time) >= rule.seconds
    return ConvergenceResult(has_converged, "converged with time limit.")
end
