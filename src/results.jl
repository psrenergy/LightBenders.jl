function get_variable_value(jump_array::Array{JuMP.VariableRef})
    jump_value = fill(NaN, size(jump_array))
    for i in eachindex(jump_array)
        if isassigned(jump_array, i)
            jump_value[i] = JuMP.value(jump_array[i])
        end
    end
    return jump_value
end

function get_variable_value(jump_array::JuMP.VariableRef)
    return JuMP.value(jump_array)
end

function get_expression_value(jump_array::Array{JuMP.AffExpr})
    jump_value = fill(NaN, size(jump_array))
    for i in eachindex(jump_array)
        if isassigned(jump_array, i)
            jump_value[i] = JuMP.value(jump_array[i])
        end
    end
    return jump_value
end

function get_expression_value(jump_array::JuMP.AffExpr)
    return JuMP.value(jump_array)
end

function get_value(jump_array)
    if first(jump_array) isa JuMP.VariableRef
        return get_variable_value(jump_array)
    elseif first(jump_array) isa JuMP.AffExpr
        return get_expression_value(jump_array)
    else
        error("Should not reach here. Array of type $(typeof(first(jump_array))).")
    end
end

function save_benders_results!(
    results::Dict{Tuple{String, Int}, Any},
    model::JuMP.Model,
    t::Int,
    scen::Int,
    num_scenarios::Int,
)
    if t == 1
        model_dict = model.obj_dict
        for (name, obj) in model_dict
            if obj isa JuMP.ConstraintRef || obj isa Array{<:JuMP.ConstraintRef}
                continue
            end
            obj_value = get_value(obj)
            for scen in 1:num_scenarios
                results[(string(name), scen)] = obj_value
            end
        end
    elseif t == 2
        model_dict = model.obj_dict
        for (name, obj) in model_dict
            if obj isa JuMP.ConstraintRef || obj isa Array{<:JuMP.ConstraintRef}
                continue
            end
            obj_value = get_value(obj)
            results[(string(name), scen)] = obj_value
        end
    end
end
