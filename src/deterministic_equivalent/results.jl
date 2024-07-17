function filter_names(var_str::Vector{String})
    var_name = String[]
    var_idx = String[]
    for i in eachindex(var_str)
        if occursin("_scen#", var_str[i])
            if occursin("[", var_str[i])
                name, idx, scen = split(var_str[i], ['[', ']'])
            else
                name, scen = split(var_str[i], "_scen#")
                idx = ""
            end
        else
            if occursin("[", var_str[i])
                name, idx = split(var_str[i], ['[', ']'])
            else
                name = var_str[i]
                idx = ""
            end
        end
        push!(var_name, name)
        push!(var_idx, idx)
    end
    return var_name, var_idx
end

function find_var_size(var_name::Vector{String}, var_idx::Vector{String})
    unique_var_name = unique(var_name)
    var_sizes = []
    for i in eachindex(unique_var_name)
        indices = findall(isequal(unique_var_name[i]), var_name)
        var_indices = var_idx[indices]
        if var_indices[1] == ""
            push!(var_sizes, [0])
            continue
        end
        indices_matrix = transpose(parse.(Int, split(var_indices[1], ',')))
        num_cols = length(indices_matrix)
        if length(indices) > 1
            for j in 2:length(indices)
                indices_matrix = vcat(indices_matrix, transpose(parse.(Int, split(var_indices[j], ','))))
            end
        end
        var_size = zeros(Int, num_cols)
        for col in 1:num_cols
            var_size[col] = maximum(indices_matrix[:, col])
        end
        push!(var_sizes, var_size)
    end
    return var_sizes
end

function begin_results(var_str::Vector{String}, num_scenarios::Int)
    var_name, var_idx = filter_names(var_str)
    unique_var_names = unique(var_name)
    var_sizes = find_var_size(var_name, var_idx)
    results = Dict{Tuple{String, Int}, Any}()
    for scen in 1:num_scenarios, var_i in eachindex(unique_var_names)
        if var_sizes[var_i] == [0]
            results[unique_var_names[var_i], scen] = 0
        else
            results[unique_var_names[var_i], scen] = zeros(var_sizes[var_i]...)
        end
    end
    return results
end

function save_deterministic_results(
    model::JuMP.Model,
    num_scenarios::Int,
)
    var = JuMP.all_variables(model)
    var_str = string.(var)
    results = begin_results(var_str, num_scenarios)
    for i in eachindex(var)
        if occursin("_scen#", var_str[i])
            if occursin("[", var_str[i])
                name, idx, scen = split(var_str[i], ['[', ']'])
                index = parse.(Int, split(idx, ','))
                scenario = parse(Int, split(scen, "_scen#")[2])
                results[name, scenario][index...] = JuMP.value(var[i])
            else
                name, scen = split(var_str[i], "_scen#")
                scenario = parse(Int, scen)
                results[name, scenario] = JuMP.value(var[i])
            end
        else # state variable results are saved in all scenarios
            if occursin("[", var_str[i])
                name, idx, scen = split(var_str[i], ['[', ']'])
                index = parse.(Int, split(idx, ','))
                for scenario in 1:num_scenarios
                    results[name, scenario][index...] = JuMP.value(var[i])
                end
            else
                name = var_str[i]
                for scenario in 1:num_scenarios
                    results[name, scenario] = JuMP.value(var[i])
                end
            end
        end
    end
    return results
end
