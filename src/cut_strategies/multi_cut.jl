"""
    CutPoolMultiCut

This is the pool that stores cuts for every scenario in the MultiCut implementation.

The cuts stored in a multicut implementation are all the cuts generated in the second stage.
"""
Base.@kwdef mutable struct CutPoolMultiCut <: AbstractCutPool
    cuts::Vector{LocalCutPool} = LocalCutPool[]
end

function number_of_cuts(pool::CutPoolMultiCut)
    return length(pool.cuts) * number_of_cuts(pool.cuts[1])
end

function store_cut!(
    pool_multicut::CutPoolMultiCut,
    local_pool::LocalCutPool,
)
    push!(pool_multicut.cuts, local_pool)
    return nothing
end

function store_cut!(
    pool::Vector{CutPoolMultiCut},
    local_cuts::LocalCutPool,
    state::Vector{Float64},
    options,
    t::Integer,
)
    store_cut!(pool[t-1], local_cuts)
    return nothing
end

function create_epigraph_multi_cut_variables!(model::JuMP.Model, policy_training_options)
    model.obj_dict[:epi_multi_cut] = Vector{JuMP.VariableRef}(undef, policy_training_options.num_scenarios)
    alphas = model.obj_dict[:epi_multi_cut]
    for scen in 1:policy_training_options.num_scenarios
        epi_multi_cut = JuMP.@variable(model, lower_bound = policy_training_options.lower_bound)
        alphas[scen] = epi_multi_cut
    end
    return alphas
end

function add_multi_cut_risk_neutral_cuts!(
    model::JuMP.Model,
    alphas::Vector{JuMP.VariableRef},
    pool::CutPoolMultiCut,
    policy_training_options,
)
    for scen in 1:policy_training_options.num_scenarios
        JuMP.set_objective_coefficient(
            model,
            alphas[scen],
            (1.0 - policy_training_options.discount_rate) / policy_training_options.num_scenarios,
        )
        for i in 1:length(pool.cuts)
            add_cut(model, alphas[scen], pool.cuts[i].coefs[scen], pool.cuts[i].rhs[scen])
        end
    end
    return nothing
end

function add_multi_cut_cvar_cuts!(
    model::JuMP.Model,
    alphas::Vector{JuMP.VariableRef},
    pool::CutPoolMultiCut,
    policy_training_options,
)
    discount_rate_multiplier = (1.0 - policy_training_options.discount_rate)
    JuMP.@variable(model, z_explicit_cvar)
    # λ * z
    JuMP.set_objective_coefficient(
        model,
        z_explicit_cvar,
        discount_rate_multiplier * (policy_training_options.risk_measure.lambda),
    )
    JuMP.@variable(model, delta_explicit_cvar[scen = 1:policy_training_options.num_scenarios] >= 0)
    for scen in 1:policy_training_options.num_scenarios
        # (1 - λ)/L * sum(alphas) 
        JuMP.set_objective_coefficient(
            model,
            alphas[scen],
            discount_rate_multiplier * (1 - policy_training_options.risk_measure.lambda) / policy_training_options.num_scenarios,
        )
        # λ / ((1 - CVaR_\alpha) * L) * sum(deltas)
        JuMP.set_objective_coefficient(
            model,
            delta_explicit_cvar[scen],
            discount_rate_multiplier *
            (policy_training_options.risk_measure.lambda) /
            ((1 - policy_training_options.risk_measure.alpha) * policy_training_options.num_scenarios),
        )
        # Add delta constraint
        JuMP.@constraint(model, delta_explicit_cvar[scen] >= alphas[scen] - z_explicit_cvar)
        # Add all cuts
        for i in 1:length(pool.cuts)
            add_cut(model, alphas[scen], pool.cuts[i].coefs[scen], pool.cuts[i].rhs[scen])
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

function get_multi_cut_future_cost(model::JuMP.Model, policy_training_options)::Float64
    if !haskey(model, :epi_multi_cut)
        return 0.0
    end
    alphas = JuMP.value.(model[:epi_multi_cut])
    if policy_training_options.risk_measure isa RiskNeutral
        return mean(alphas)
    elseif policy_training_options.risk_measure isa CVaR
        discount_rate_multiplier = (1.0 - policy_training_options.discount_rate)
        z_explicit_cvar = JuMP.value(model[:z_explicit_cvar])
        delta_explicit_cvar = JuMP.value.(model[:delta_explicit_cvar])
        fcf = z_explicit_cvar * discount_rate_multiplier * (policy_training_options.risk_measure.lambda)
        for scen in 1:policy_training_options.num_scenarios
            fcf += alphas[scen] * discount_rate_multiplier * (1 - policy_training_options.risk_measure.lambda) / policy_training_options.num_scenarios
            fcf += delta_explicit_cvar[scen] * discount_rate_multiplier * (policy_training_options.risk_measure.lambda) / ((1 - policy_training_options.risk_measure.alpha) * policy_training_options.num_scenarios)
        end
        return fcf
    end
end
