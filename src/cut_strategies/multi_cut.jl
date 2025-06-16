"""
    CutPoolMultiCut

This is the pool that stores cuts for every scenario in the MultiCut implementation.

The cuts stored in a multicut implementation are all the cuts generated in the second stage.
"""
Base.@kwdef mutable struct CutPoolMultiCut <: AbstractCutPool
    cuts::Vector{CutPoolSingleCut}
    function CutPoolMultiCut(options)
        return new([CutPoolSingleCut(manager = CutRelaxationData(options.cut_relaxation)) for _ in 1:options.num_scenarios],)
    end
end

function number_of_cuts(pool::CutPoolMultiCut)
    return length(pool.cuts) * number_of_cuts(pool.cuts[1])
end

function store_cut!(
    pool_multicut::CutPoolMultiCut,
    local_pool::LocalCutPool,
)
    for scen in 1:length(pool_multicut.cuts)
        store_cut!(
            pool_multicut.cuts[scen],
            local_pool.coefs[scen],
            local_pool.state[scen],
            local_pool.rhs[scen],
            local_pool.obj[scen],
        )
    end
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

function create_epigraph_multi_cut_variables!(model::JuMP.Model, pool, policy_training_options)
    model.obj_dict[:epi_multi_cut] = Vector{JuMP.VariableRef}(undef, policy_training_options.num_scenarios)
    alphas = model.obj_dict[:epi_multi_cut]
    for scen in 1:policy_training_options.num_scenarios
        epi_multi_cut = JuMP.@variable(model, lower_bound = policy_training_options.lower_bound)
        alphas[scen] = epi_multi_cut
        JuMP.set_objective_coefficient(
            model,
            alphas[scen],
            (1.0 - policy_training_options.discount_rate) / policy_training_options.num_scenarios,
        )
    end
    if policy_training_options.risk_measure isa CVaR
        JuMP.@variable(model, z_explicit_cvar)
        JuMP.@variable(model, delta_explicit_cvar[scen = 1:policy_training_options.num_scenarios] >= 0)
        discount_rate_multiplier = (1.0 - policy_training_options.discount_rate)
        z_explicit_cvar = model[:z_explicit_cvar]
        delta_explicit_cvar = model[:delta_explicit_cvar]
        # λ * z
        JuMP.set_objective_coefficient(
            model,
            z_explicit_cvar,
            discount_rate_multiplier * (policy_training_options.risk_measure.lambda),
        )
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
        end
    end
    for scen in 1:policy_training_options.num_scenarios
        pool_cuts_scen = pool.cuts[scen]
        pool_cuts_scen.manager.epigraph_variable = alphas[scen]
        pool_cuts_scen.manager.epigraph_value = Inf
    end

    return nothing
end

function add_all_cuts!(model::JuMP.Model, pool::CutPoolMultiCut, policy_training_options)
    for l in 1:length(pool.cuts)
        add_all_cuts!(model, pool.cuts[l], policy_training_options)
    end
    return nothing
end

function add_incremental_cut!(model::JuMP.Model, pool::CutPoolMultiCut, policy_training_options)
    for l in 1:length(pool.cuts)
        add_incremental_cut!(model, pool.cuts[l], policy_training_options)
    end
    return nothing
end

function add_initial_cuts!(model::JuMP.Model, pool::CutPoolMultiCut, policy_training_options)
    for l in 1:length(pool.cuts)
        add_initial_cuts!(model, pool.cuts[l], policy_training_options)
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
            fcf +=
                alphas[scen] * discount_rate_multiplier * (1 - policy_training_options.risk_measure.lambda) /
                policy_training_options.num_scenarios
            fcf +=
                delta_explicit_cvar[scen] * discount_rate_multiplier * (policy_training_options.risk_measure.lambda) /
                ((1 - policy_training_options.risk_measure.alpha) * policy_training_options.num_scenarios)
        end
        return fcf
    end
end

function cut_relaxation_inner!(model::JuMP.Model, pool::CutPoolMultiCut)
    has_violation = false
    for l in 1:length(pool.cuts)
        @timeit_debug to_train "Cut Relaxation Inner" has_violation_l = cut_relaxation_inner!(model, pool.cuts[l])
        if has_violation_l
            has_violation = true
            break
        end
    end
    return has_violation
end

function update_epigraph_value!(pool::CutPoolMultiCut)
    for l in 1:length(pool.cuts)
        data = pool.cuts[l].manager
        if isnothing(data.epigraph_variable)
            continue
        end
        data.epigraph_value = JuMP.value.(data.epigraph_variable)
    end
    return nothing
end

function reset_cuts!(model::JuMP.Model, pool::CutPoolMultiCut, progress)
    for l in 1:length(pool.cuts)
        reset_cuts!(model, pool.cuts[l], progress)
    end
    return nothing
end
