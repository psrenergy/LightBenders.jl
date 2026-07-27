"""
    AbstractRegularization

Abstract type to hold regularization implementations for the first stage problem.
"""
abstract type AbstractRegularization end

"""
    NoRegularization

Default option: the state passed to the second stage is the optimal solution of the
first stage problem.
"""
struct NoRegularization <: AbstractRegularization end

"""
    LevelSetRegularization

Interior level-set regularization from Pecci & Jenkins, "Regularized Benders
Decomposition for High Performance Capacity Expansion Models" (IEEE Transactions on
Power Systems, 2025).

After the first stage problem is solved to obtain the lower bound `LB`, the first
stage problem is re-solved with the level-set constraint

    objective <= LB + alpha * (UB - LB)

and a zero objective, where `UB` is the best upper bound found so far. Solving this
feasibility problem with an interior point method (without crossover) selects a trial
state in the interior of the level set, which avoids the extreme solutions that slow
down cutting-plane convergence.

# Fields

  - `alpha::Float64`: level-set parameter in (0, 1). Controls how far above the lower
    bound the trial solutions are allowed to be. Defaults to 0.5.
  - `optimizer_attributes::Vector{Pair{String, Any}}`: solver attributes applied only
    during the level-set solve and restored afterwards. Use them to select a barrier /
    interior point algorithm without crossover, e.g. `["solver" => "ipm",
    "run_crossover" => "off"]` for HiGHS or `["DEFAULTALG" => 4, "CROSSOVER" => 0]`
    for Xpress.
"""
mutable struct LevelSetRegularization <: AbstractRegularization
    alpha::Float64
    optimizer_attributes::Vector{Pair{String, Any}}
    function LevelSetRegularization(;
        alpha::Real = 0.5,
        optimizer_attributes::Vector = Pair{String, Any}[],
    )
        if !(0.0 < alpha < 1.0)
            throw(ArgumentError("LevelSetRegularization alpha must be in (0, 1)."))
        end
        return new(alpha, convert(Vector{Pair{String, Any}}, optimizer_attributes))
    end
end

"""
    has_integrality(model::JuMP.Model)

Return whether the model currently enforces integrality on any variable. Used to skip
the level-set regularization in the integer phase of the training (following the
reference paper, which regularizes only the continuous relaxation).
"""
function has_integrality(model::JuMP.Model)
    return JuMP.num_constraints(model, JuMP.VariableRef, MOI.Integer) > 0 ||
           JuMP.num_constraints(model, JuMP.VariableRef, MOI.ZeroOne) > 0
end

"""
    solve_level_set_problem!(model, policy_training_options, lower_bound, best_upper_bound)

Solve the interior level-set problem on the first stage `model`, which must have just
been solved to optimality (its optimal value being `lower_bound`).

Returns a tuple `(state, first_stage_cost)` where `state` is the trial state to pass
to the second stage and `first_stage_cost` is the first stage cost (excluding the
future cost) evaluated at the trial state. Returns `nothing` if the level-set solve
fails, in which case callers should fall back to the unregularized solution.

The model's objective and optimizer attributes are restored before returning.
"""
function solve_level_set_problem!(
    model::JuMP.Model,
    policy_training_options,
    lower_bound::Float64,
    best_upper_bound::Float64,
)
    regularization = policy_training_options.regularization::LevelSetRegularization
    objfun = JuMP.objective_function(model)
    level = lower_bound + regularization.alpha * (best_upper_bound - lower_bound)
    level_set_constraint = JuMP.@constraint(model, objfun <= level)
    JuMP.@objective(model, Min, 0)
    previous_attributes = Pair{String, Any}[]
    for (key, value) in regularization.optimizer_attributes
        push!(previous_attributes, key => JuMP.get_attribute(model, key))
        JuMP.set_attribute(model, key, value)
    end
    JuMP.optimize!(model)
    result = nothing
    if JuMP.termination_status(model) == MOI.OPTIMAL && JuMP.has_values(model)
        state = get_state(model)
        first_stage_cost =
            JuMP.value(objfun) - get_future_cost(model, policy_training_options)
        result = (state, first_stage_cost)
    else
        @warn(
            "Level-set problem finished with termination status " *
            "$(JuMP.termination_status(model)). Falling back to the unregularized " *
            "first stage solution."
        )
    end
    for (key, value) in previous_attributes
        JuMP.set_attribute(model, key, value)
    end
    JuMP.delete(model, level_set_constraint)
    JuMP.set_objective_function(model, objfun)
    return result
end
