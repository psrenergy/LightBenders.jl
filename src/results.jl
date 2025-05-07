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
            obj_value = JuMP.value.(obj)
            for s in 1:num_scenarios
                results[string(name), s] = obj_value
            end
        end
    elseif t == 2
        model_dict = model.obj_dict
        for (name, obj) in model_dict
            obj_value = JuMP.value.(obj)
            results[string(name), scen] = obj_value
        end
    end
    return nothing
end
