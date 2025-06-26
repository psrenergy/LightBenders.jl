"""
    CutPoolSingleCut

This is the pool that stores cuts for every scenario in the SingleCut implementation.

The cuts stored in a single cut implementation are the average cuts of all cuts generated in the second stage.

The first dimension is scenarios and the second is states
"""
Base.@kwdef mutable struct CutPoolSingleCut <: AbstractCutPool
    coefs::Vector{Vector{Float64}} = Vector{Float64}[]
    state::Vector{Vector{Float64}} = Vector{Float64}[]
    rhs::Vector{Float64} = Float64[]
    obj::Vector{Float64} = Float64[]
    manager::CutRelaxationData
end

function CutPoolSingleCut(options)
    manager = CutRelaxationData(options.cut_relaxation)
    return CutPoolSingleCut(manager = manager)
end

function number_of_cuts(pool::CutPoolSingleCut)
    return length(pool.rhs)
end

function cut_is_different(
    pool::Union{CutPoolSingleCut, LocalCutPool},
    coefs::Vector{Float64},
    state::Vector{Float64},
    rhs::Float64,
    obj::Float64,
)
    return true
    if isempty(pool)
        return true
    end
    ATOL = 1e-26
    # First check only the rhs and obj
    for i in number_of_cuts(pool):-1:1
        if !isapprox(pool.rhs[i], rhs; atol = ATOL)
            return true
        end
        if !isapprox(pool.obj[i], obj; atol = ATOL)
            return true
        end
    end

    # Then check the coefficients
    for i in 1:number_of_cuts(pool)
        for j in 1:size(pool.coefs, 1)
            if !isapprox(pool.coefs[j, i], coefs[j]; atol = ATOL)
                return true
            end
        end
    end

    # Then check the states
    for i in number_of_cuts(pool):-1:1
        if !isapprox(pool.state[i], state; atol = ATOL)
            return true
        end
    end

    return false
end

function store_cut!(
    pool::CutPoolSingleCut,
    coefs::Vector{Float64},
    state::Vector{Float64},
    rhs::Float64,
    obj::Float64,
)
    if !cut_is_different(pool, coefs, state, rhs, obj)
        return nothing
    end
    push!(pool.coefs, coefs)
    push!(pool.state, state)
    push!(pool.rhs, rhs)
    push!(pool.obj, obj)
    return nothing
end

function store_cut!(
    pool::Vector{CutPoolSingleCut},
    local_cuts::LocalCutPool,
    state::Vector{Float64},
    options,
    t::Integer,
)
    if isa(options.risk_measure, RiskNeutral)
        return risk_neutral_single_cut!(pool, local_cuts, state, options, t)
    elseif isa(options.risk_measure, CVaR)
        return cvar_single_cut!(pool, local_cuts, state, options, t)
    else
        error("Risk measure not implemented.")
    end
end

function risk_neutral_single_cut!(
    pool::Vector{CutPoolSingleCut},
    local_cuts::LocalCutPool,
    state::Vector{Float64},
    options,
    t::Integer,
)
    num_local_cuts = length(local_cuts.obj)
    obj = mean(local_cuts.obj)
    rhs = mean(local_cuts.rhs)
    coefs = zeros(Float64, length(local_cuts.coefs[1]))
    for i in eachindex(coefs)
        coefs[i] = sum(local_cuts.coefs[j][i] for j in 1:num_local_cuts) / num_local_cuts
    end
    store_cut!(pool[t-1], coefs, state, rhs, obj)
    return nothing
end

function cvar_single_cut!(
    pool::Vector{CutPoolSingleCut},
    local_cuts::LocalCutPool,
    state::Vector{Float64},
    options,
    t::Int,
)
    weights = build_cvar_weights(local_cuts.obj, options.risk_measure.alpha, options.risk_measure.lambda)
    obj = dot(weights, local_cuts.obj)
    rhs = dot(weights, local_cuts.rhs)
    coefs = zeros(Float64, length(local_cuts.coefs[1]))
    for j in eachindex(weights)
        coefs .+= weights[j] .* local_cuts.coefs[j]
    end
    store_cut!(pool[t-1], coefs, state, rhs, obj)
    return nothing
end

function create_epigraph_single_cut_variables!(model::JuMP.Model, pool, policy_training_options)
    JuMP.@variable(model, epi_single_cut, lower_bound = policy_training_options.lower_bound)
    JuMP.set_objective_coefficient(model, epi_single_cut, (1.0 - policy_training_options.discount_rate))
    pool.manager.epigraph_variable = epi_single_cut
    pool.manager.epigraph_value = Inf
    return nothing
end

function add_all_cuts!(model::JuMP.Model, pool::CutPoolSingleCut, policy_training_options)
    alpha = pool.manager.epigraph_variable::JuMP.VariableRef
    for i in 1:number_of_cuts(pool)
        add_cut(model, alpha, pool.coefs[i], pool.rhs[i])
    end
    return nothing
end

function add_incremental_cut!(model::JuMP.Model, pool, policy_training_options)
    alpha = pool.manager.epigraph_variable::JuMP.VariableRef
    if number_of_cuts(pool) == 0
        return nothing
    end
    add_cut(model, alpha, pool.coefs[end], pool.rhs[end])
    return nothing
end

function add_initial_cuts!(model::JuMP.Model, pool::CutPoolSingleCut, policy_training_options)
    data = pool.manager
    if !data.options.warmstart
        return nothing
    end
    alpha = data.epigraph_variable::JuMP.VariableRef
    for i in max(1, number_of_cuts(pool) - data.options.warmstart_size + 1):number_of_cuts(pool)
        add_cut(model, alpha, pool.coefs[i], pool.rhs[i])
    end
    return nothing
end

function add_violations!(model::JuMP.Model, pool::CutPoolSingleCut, violations)
    alpha = pool.manager.epigraph_variable::JuMP.VariableRef
    for i in violations
        if i == 0 || isempty(pool.coefs)
            continue
        end
        cref = add_cut(model, alpha, pool.coefs[i], pool.rhs[i])
        push!(pool.manager.added_cuts_ref, cref)
    end
    return nothing
end

function get_violations(
    pool::CutPoolSingleCut,
)
    data = pool.manager
    added = data.added_cuts
    cache = data.violation_cache
    if length(cache) != number_of_cuts(pool)
        resize!(cache, number_of_cuts(pool))
    end

    cache .+= pool.rhs .- data.epigraph_value
    k = min(data.options.step_size, number_of_cuts(pool))
    tol = data.options.tol

    candidate_cache = data.candidate_cache
    if length(candidate_cache) != number_of_cuts(pool)
        resize!(candidate_cache, number_of_cuts(pool))
    end

    candidates =
        partialsortperm!(candidate_cache, cache, 1:k, rev = true, initialized = false)

    has_violation = false
    for i in k:-1:1
        c = candidates[i]
        v = cache[c]
        if v >= tol && !(c in added)
            push!(added, c)
            has_violation = true
        else
            candidates[i] = 0
        end
    end

    return has_violation, candidates
end

function cut_relaxation_inner!(
    model::JuMP.Model,
    pool::CutPoolSingleCut,
)
    @timeit_debug to_train "Get Violations" has_violation, violations = get_violations(pool)
    if has_violation
        @timeit_debug to_train "Add Violations" add_violations!(model, pool, violations)
    end
    return has_violation
end

function update_epigraph_value!(pool::CutPoolSingleCut)
    data = pool.manager
    if isnothing(data.epigraph_variable)
        return nothing
    end
    data.epigraph_value = JuMP.value(data.epigraph_variable)
    return nothing
end

function reset_cuts!(model::JuMP.Model, pool::CutPoolSingleCut, progress)
    # Reset cuts at every reset_step iterations
    reset_step = pool.manager.options.reset_step
    if mod(progress.current_iteration, reset_step) != 0
        return nothing
    end
    @timeit_debug to_train "Reset Cut Relaxation Data" reset_cut_relaxation_data!(model, pool.manager)
    return nothing
end

function get_single_cut_future_cost(model::JuMP.Model)::Float64
    if !haskey(model, :epi_single_cut)
        return 0.0
    end
    alpha = model[:epi_single_cut]::JuMP.VariableRef
    return JuMP.value(alpha)
end
