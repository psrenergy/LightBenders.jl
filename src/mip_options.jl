"""
    MIPOptions

Options for the integer phase of the Benders master.

- `run_mip_after_iteration`: iterations solved with the master's integrality
  relaxed before it is restored (0 = never relax).
- `dynamic_gap`: when the master is a MIP, set its relative gap tolerance each
  iteration to `gap_fraction` times the current Benders relative gap, clamped
  to `[min_gap, max_gap]`. Early iterations then stop at (or near) the root
  node; the tolerance tightens only as the Benders bounds close.
- `lower_bound_from_bound`: use the master's proven objective bound (`JuMP.objective_bound`)
  as the Benders lower bound instead of the incumbent's objective. Required for
  a valid lower bound whenever the master stops with a nonzero MIP gap.
"""
Base.@kwdef mutable struct MIPOptions
    run_mip_after_iteration::Int = 50
    dynamic_gap::Bool = false
    gap_fraction::Float64 = 0.5
    min_gap::Float64 = 1e-4
    max_gap::Float64 = 0.1
    lower_bound_from_bound::Bool = false
end

"""
    master_gap_target(progress, mip_options) -> Float64

Relative MIP gap to request from the master this iteration under `dynamic_gap`:
`gap_fraction` times the current Benders relative gap (best upper bound vs the
last lower bound), clamped to `[min_gap, max_gap]`. Falls back to `max_gap`
before an upper bound exists.
"""
function master_gap_target(progress, mip_options::MIPOptions)
    ub = progress.best_UB
    lb = progress.current_iteration > 1 ? progress.LB[progress.current_iteration - 1] : -Inf
    if !isfinite(ub) || !isfinite(lb) || ub == 0
        return mip_options.max_gap
    end
    gap = abs(ub - lb) / abs(ub)
    return clamp(mip_options.gap_fraction * gap, mip_options.min_gap, mip_options.max_gap)
end

function set_master_gap!(model::JuMP.Model, gap::Float64)
    JuMP.set_attribute(model, MOI.RelativeGapTolerance(), gap)
    return nothing
end

"""
    master_lower_bound_value(model, mip_options, integer_active) -> Float64

Contribution of the master solve to the Benders lower bound: the proven
objective bound when the master is an integer program solved with a gap and
`lower_bound_from_bound` is set, the objective value otherwise.
"""
function master_lower_bound_value(model::JuMP.Model, mip_options::MIPOptions, integer_active::Bool)
    if integer_active && mip_options.lower_bound_from_bound
        bound = try
            JuMP.objective_bound(model)
        catch
            NaN
        end
        isfinite(bound) && return bound
    end
    return JuMP.objective_value(model)
end
