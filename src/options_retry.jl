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
end

function store_retry_data(model, options)
    model.ext[:retry_optimize_options] = options.retry_optimize
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
    # Then try changing options
    for options in data
        current = Pair{String, Any}[]
        for (key, value) in options
            temp = get_attribute(model, key)
            push!(current, key => temp)
            set_attribute(model, key, value)
        end
        JuMP.optimize!(model)
        status = JuMP.termination_status(model)
        for (key, value) in current
            set_attribute(model, key, value)
        end
        if status == MOI.OPTIMAL
            break
        end
    end
    return nothing
end
