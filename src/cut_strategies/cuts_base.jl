"""
    CutStragey

Enum type to store all possible cut strategy implmentations.
"""
@enumx CutStrategy begin
    SingleCut = 0
    MultiCut = 1
end

"""
    AbstractCutPool

Abstract type to hold various implementations of cut pools. A cut pool is a data structure to store cuts.
"""
abstract type AbstractCutPool end

"""
    initialize_cut_pool(options)

Create a cut pool for every stage of the problem.
"""
function initialize_cut_pool(options)
    num_stages = 2
    if options.cut_strategy == CutStrategy.SingleCut
        return [LightBenders.CutPoolSingleCut() for _ in 1:num_stages]
    elseif options.cut_strategy == CutStrategy.MultiCut
        return [LightBenders.CutPoolMultiCut() for _ in 1:num_stages]
    end
    error("Not implemented.")
    return nothing
end

"""
"""
function get_future_cost(model::JuMP.Model, policy_training_options)::Float64
    if policy_training_options.cut_strategy == CutStrategy.SingleCut
        return LightBenders.get_single_cut_future_cost(model)
    else
        return LightBenders.get_multi_cut_future_cost(model, policy_training_options)
    end
end

"""
    get_cut(model, states)

Return the coefficients, rhs and obj of a calculated cut.
"""
function get_cut(model, states)
    cache = model.ext[:second_stage_state]::StateCache
    coefs = Vector{Float64}(undef, length(cache.variables))
    obj = JuMP.objective_value(model)
    rhs = obj
    for i in eachindex(cache.variables)
        if JuMP.is_parameter(cache.variables[i])
            coefs[i] = JuMP.dual(ParameterRef(cache.variables[i]))
        else
            coefs[i] = JuMP.reduced_cost(cache.variables[i])
        end
        # coefs[i] = JuMP.dual(JuMP.FixRef(cache.variables[i]))
        rhs -= coefs[i] * states[i]
    end
    return truncate_small_numbers.(coefs), truncate_small_numbers(rhs), truncate_small_numbers(obj)
end

"""
    add_cut(model::JuMP.Model, epigraph_variable::JuMP.VariableRef, coefs::Vector{T}, rhs::T, sense::MOI.OptimizationSense) where T <: Real

Add a cut to a Model and return the constraint reference.
For minimization problems, cuts are lower bounds: alpha >= rhs + coefs * x
For maximization problems, cuts are upper bounds: alpha <= rhs + coefs * x
"""
function add_cut(model::JuMP.Model, epigraph_variable::JuMP.VariableRef, coefs::Vector{T}, rhs::T, sense::MOI.OptimizationSense) where {T <: Real}
    alpha = epigraph_variable
    cache = model.ext[:first_stage_state]::StateCache
    if is_minimization(sense)
        cref = @constraint(model, alpha >= rhs + dot(coefs, cache.variables))
    else
        cref = @constraint(model, alpha <= rhs + dot(coefs, cache.variables))
    end
    return cref
end

function create_epigraph_variables!(model::JuMP.Model, policy_training_options, sense::MOI.OptimizationSense)
    if policy_training_options.cut_strategy == CutStrategy.SingleCut
        return create_epigraph_single_cut_variables!(model, policy_training_options, sense)
    elseif policy_training_options.cut_strategy == CutStrategy.MultiCut
        return create_epigraph_multi_cut_variables!(model, policy_training_options, sense)
    end
    error("Not implemented.")
    return nothing
end

function number_of_cuts(pool::Nothing)
    return 0
end
