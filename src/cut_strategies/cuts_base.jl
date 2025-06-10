"""
    CutStragey

Enum type to store all possible cut strategy implmentations.
"""
@enumx CutStrategy begin
    SingleCut = 0
    MultiCut = 1
end

const ScalarAffineFunction_GreaterThan = JuMP.ConstraintRef{
    JuMP.Model,
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}},
    JuMP.ScalarShape,
}

mutable struct CutRelaxationData
    violation_cache::Vector{Float64}
    candidate_cache::Vector{Int}
    added_cuts::Set{Int}
    added_cuts_ref::Vector{ScalarAffineFunction_GreaterThan}
    options::CutRelaxationOptions
    epigraph_variable::Union{JuMP.VariableRef, Nothing}
    epigraph_value::Float64
    function CutRelaxationData(cut_relaxation_options)
        cut_relaxation_data = new()
        cut_relaxation_data.violation_cache = zeros(Float64, 0)
        cut_relaxation_data.candidate_cache = zeros(Int, 0)
        cut_relaxation_data.added_cuts = Set{Int}()
        sizehint!(cut_relaxation_data.added_cuts, 30)
        cut_relaxation_data.added_cuts_ref = ScalarAffineFunction_GreaterThan[]
        sizehint!(cut_relaxation_data.added_cuts_ref, 30)
        cut_relaxation_data.options = cut_relaxation_options
        cut_relaxation_data.epigraph_variable = nothing
        cut_relaxation_data.epigraph_value = 0.0
        return cut_relaxation_data
    end
end

function reset_cut_relaxation_data!(model::JuMP.Model, data::CutRelaxationData)
    N = length(data.added_cuts)
    if N == 0
        return nothing
    end
    JuMP.delete(model, data.added_cuts_ref)
    resize!(data.added_cuts_ref, 0)
    empty!(data.added_cuts)
    data.epigraph_value = Inf
    return nothing
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
        return [LightBenders.CutPoolSingleCut(options) for _ in 1:num_stages]
    elseif options.cut_strategy == CutStrategy.MultiCut
        return [LightBenders.CutPoolMultiCut(options) for _ in 1:num_stages]
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

function initialize_cuts!(model::JuMP.Model, pool, options)

    if options.cut_relaxation.active
        add_initial_cuts!(model, pool, options)
    else
        add_all_cuts!(model, pool, options)
    end
    return nothing
end

function simple_optimize_first_stage(model::JuMP.Model, policy_training_options, progress)
    optimize_with_retry(model)
    treat_termination_status(model, policy_training_options, 1, progress.current_iteration)
end

function optimize_first_stage(first_stage_model::JuMP.Model, pool, policy_training_options, progress)
    store_retry_data(first_stage_model, policy_training_options)
    if policy_training_options.cut_relaxation.active
        cut_relaxation_optimize(first_stage_model, policy_training_options, pool, progress)
    else
        simple_optimize_first_stage(first_stage_model, policy_training_options, progress)
    end
    return nothing
end


function cut_relaxation_optimize(model::JuMP.Model, policy_training_options, pool, progress)
    optimize_with_retry(model)
    treat_termination_status(model, policy_training_options, 1, progress.current_iteration)

    cut_iter = 0
    while true
        cut_iter += 1
        update_epigraph_value!(pool)

        has_violation = cut_relaxation_inner!(model, pool)

        if !has_violation
            break
        end

        optimize_with_retry(model)
        treat_termination_status(model, policy_training_options, 1, progress.current_iteration)
    end

    return nothing
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
    add_cut(model::JuMP.Model, epigraph_variable::JuMP.VariableRef, coefs::Vector{T}, rhs::T) where T <: Real

Add a cut to a Model and return the constraint reference.
"""
function add_cut(model::JuMP.Model, epigraph_variable::JuMP.VariableRef, coefs::Vector{T}, rhs::T) where {T <: Real}
    alpha = epigraph_variable
    cache = model.ext[:first_stage_state]::StateCache
    cref = @constraint(model, alpha >= rhs + dot(coefs, cache.variables))
    return cref
end

function create_epigraph_variables!(model::JuMP.Model, pool, policy_training_options)
    if policy_training_options.cut_strategy == CutStrategy.SingleCut
        return create_epigraph_single_cut_variables!(model, pool, policy_training_options)
    elseif policy_training_options.cut_strategy == CutStrategy.MultiCut
        return create_epigraph_multi_cut_variables!(model, pool, policy_training_options)
    end
    error("Not implemented.")
    return nothing
end

function number_of_cuts(pool::Nothing)
    return 0
end
