"""
    RetryOptimizeOptions

Options for retrying optimization with different options.

example
retry_optimize = LightBenders.RetryOptimizeOptions(
callback = function retry(model)
HiGHS.Highs_clearSolver(backend(model).optimizer)
return nothing
end,
data = Vector{Pair{String,Any}}[
Pair{String,Any}[
"presolve" => "off",
"solver" => "simplex",
"simplex_strategy" => 1,
],
Pair{String,Any}[
"presolve" => "on",
"solver" => "simplex",
"simplex_strategy" => 4,
],
Pair{String,Any}[
"presolve" => "on",
"solver" => "ipm",
],
]
),
"""
Base.@kwdef mutable struct RetryOptimizeOptions
    callback::Union{Function, Nothing} = nothing
    data::Vector{Vector{Pair{String, Any}}} = Vector{Pair{String, Any}}[]
    # Optional ladder used for second-stage (scenario) models instead of `data`.
    # Lets the caller allow e.g. crossover on the small subproblems while the
    # large master stays on barrier only.
    second_stage_data::Union{Nothing, Vector{Vector{Pair{String, Any}}}} = nothing
end

function store_retry_data(model, options; second_stage::Bool = false)
    retry = options.retry_optimize
    if second_stage && retry !== nothing && retry.second_stage_data !== nothing
        retry = RetryOptimizeOptions(; callback = retry.callback, data = retry.second_stage_data)
    end
    model.ext[:retry_optimize_options] = retry
    return nothing
end

function optimize_with_retry(model)::Nothing
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    if status == MOI.OPTIMAL
        return nothing
    end
    if !haskey(model.ext, :retry_optimize_options)
        return nothing
    end
    retry_options = model.ext[:retry_optimize_options]::RetryOptimizeOptions
    data = retry_options.data
    callback = retry_options.callback
    # First try only with callback
    if callback !== nothing
        callback(model)
        JuMP.optimize!(model)
        status = JuMP.termination_status(model)
        if status == MOI.OPTIMAL
            return nothing
        end
    end
    # Then try changing options. When an attempt succeeds, keep its settings:
    # restoring attributes after the solve invalidates the solution just
    # obtained (JuMP then raises OptimizeNotCalled on any query), and the
    # settings that worked are the ones the next solve of this model wants.
    for (i, options) in enumerate(data)
        current = Pair{String, Any}[]
        for (key, value) in options
            temp = get_attribute(model, key)
            push!(current, key => temp)
            set_attribute(model, key, value)
        end
        JuMP.optimize!(model)
        status = JuMP.termination_status(model)
        if status == MOI.OPTIMAL
            return nothing
        end
        # After the last rung keep the attributes: touching them would mark the
        # model as modified and turn the real failure status into
        # OPTIMIZE_NOT_CALLED, hiding what went wrong.
        if i < length(data)
            for (key, value) in current
                set_attribute(model, key, value)
            end
        end
    end
    @warn "All retry attempts failed; last termination status: $(JuMP.termination_status(model))"
    return nothing
end
