"""
    CutPoolMultiCut

This is the pool that stores cuts for every scenario in the MultiCut implementation.

The cuts stored in a multicut implementation are all the cuts generated in the backward phase.
"""
Base.@kwdef mutable struct CutPoolMultiCut <: AbstractCutPool
    cuts::Vector{CutPoolBackwardPhase} = CutPoolBackwardPhase[]
end

function number_of_cuts(pool::CutPoolMultiCut)
    return length(pool.cuts) * number_of_cuts(pool.cuts[1])
end

function store_cut!(
    pool_multicut::CutPoolMultiCut, 
    pool_backward::CutPoolBackwardPhase
)
    push!(pool_multicut.cuts, pool_backward)
    return nothing
end

function store_cut!(
    pool::Vector{CutPoolMultiCut}, 
    backward_cuts::CutPoolBackwardPhase,
    state::Vector{Float64},
    options,
    t::Integer
)
    store_cut!(pool[t-1], backward_cuts)
end

function create_epigraph_multi_cut_variables!(model::JuMP.Model, policy_training_options)
    model.obj_dict[:epi_multi_cut] = Vector{JuMP.VariableRef}(undef, policy_training_options.num_openings)
    alphas = model.obj_dict[:epi_multi_cut]
    for l in 1:policy_training_options.num_openings
        epi_multi_cut = JuMP.@variable(model, lower_bound = policy_training_options.lower_bound)
        alphas[l] = epi_multi_cut
    end
    return alphas
end

function add_multi_cut_risk_neutral_cuts!(
    model::JuMP.Model, 
    alphas::Vector{JuMP.VariableRef},
    pool::CutPoolMultiCut, 
    policy_training_options
)
    for l in 1:policy_training_options.num_openings
        JuMP.set_objective_coefficient(
            model, 
            alphas[l], 
            (1.0 - policy_training_options.discount_rate) / policy_training_options.num_openings
        )
        for i in 1:length(pool.cuts)
            add_cut(model, alphas[l], pool.cuts[i].coefs[l], pool.cuts[i].rhs[l])
        end
    end
    return nothing
end

function add_multi_cut_cvar_cuts!(
    model::JuMP.Model, 
    alphas::Vector{JuMP.VariableRef},
    pool::CutPoolMultiCut, 
    policy_training_options
)
    discount_rate_multiplier = (1.0 - policy_training_options.discount_rate)
    JuMP.@variable(model, z_explicit_cvar)
    # λ * z
    JuMP.set_objective_coefficient(
        model, 
        z_explicit_cvar, 
        discount_rate_multiplier * (policy_training_options.risk_measure.lambda)
    )
    JuMP.@variable(model, delta_explicit_cvar[l = 1:policy_training_options.num_openings] >= 0)
    for l in 1:policy_training_options.num_openings
        # (1 - λ)/L * sum(alphas) 
        JuMP.set_objective_coefficient(
            model, 
            alphas[l], 
            discount_rate_multiplier * (1 - policy_training_options.risk_measure.lambda) / policy_training_options.num_openings
        )
        # λ / ((1 - CVaR_\alpha) * L) * sum(deltas)
        JuMP.set_objective_coefficient(
            model, 
            delta_explicit_cvar[l], 
            discount_rate_multiplier * 
            (policy_training_options.risk_measure.lambda) / ((1 - policy_training_options.risk_measure.alpha) * policy_training_options.num_openings)
        )
        # Add delta constraint
        JuMP.@constraint(model, delta_explicit_cvar[l] >= alphas[l] - z_explicit_cvar)
        # Add all cuts
        for i in 1:length(pool.cuts)
            add_cut(model, alphas[l], pool.cuts[i].coefs[l], pool.cuts[i].rhs[l])
        end
    end
end

function add_all_cuts!(model::JuMP.Model, pool::CutPoolMultiCut, policy_training_options)
    if isempty(pool.cuts)
        return nothing
    end
    # Add the epigraph variables. These variables must be common to every risk adjusted implementation.
    alphas = create_epigraph_multi_cut_variables!(model, policy_training_options)

    if isa(policy_training_options.risk_measure, RiskNeutral)
        add_multi_cut_risk_neutral_cuts!(model, alphas, pool, policy_training_options)
    elseif isa(policy_training_options.risk_measure, CVaR)
        add_multi_cut_cvar_cuts!(model, alphas, pool, policy_training_options)
    end
    return nothing
end

function get_multi_cut_future_cost(model::JuMP.Model)::Float64
    if !haskey(model, :epi_multi_cut)
        return 0.0
    end
    alphas = model[:epi_multi_cut]::Vector{JuMP.VariableRef}
    return mean(JuMP.value.(alphas))
end