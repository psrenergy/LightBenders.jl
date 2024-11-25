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
end

function number_of_cuts(pool::CutPoolSingleCut)
    return length(pool.rhs)
end

function store_cut!(
    pool::CutPoolSingleCut,
    coefs::Vector{Float64},
    state::Vector{Float64},
    rhs::Float64,
    obj::Float64,
)
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

function add_all_cuts!(model::JuMP.Model, pool::CutPoolSingleCut, policy_training_options)
    JuMP.@variable(model, epi_single_cut >= policy_training_options.lower_bound)
    JuMP.set_objective_coefficient(model, epi_single_cut, (1.0 - policy_training_options.discount_rate))
    for i in 1:number_of_cuts(pool)
        add_cut(model, epi_single_cut, pool.coefs[i], pool.rhs[i])
    end
    return nothing
end

function get_single_cut_future_cost(model::JuMP.Model)::Float64
    if !haskey(model, :epi_single_cut)
        return 0.0
    end
    alpha = model[:epi_single_cut]::JuMP.VariableRef
    return JuMP.value(alpha)
end
