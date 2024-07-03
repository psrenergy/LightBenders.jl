"""
"""
struct StateDef
    name::Symbol
    first::Int
    len::Int
end

"""
"""
Base.@kwdef mutable struct StateCache
    variables::Vector{JuMP.VariableRef} = JuMP.VariableRef[]
    state::Vector{StateDef} = StateDef[]
end

"""
"""
function store_state_cache(state_cache_in, state_cache_out, model, t)
    in_cache = model.ext[:state_in]::StateCache
    out_cache = model.ext[:state_out]::StateCache
    state_cache_in[t] = in_cache
    state_cache_out[t] = out_cache
    return nothing
end

"""
"""
function set_state_input(model, name::Symbol, variables)
    cache = model.ext[:state_in]::StateCache
    init_state(cache, name, variables)
    return nothing
end

"""
"""
function set_state_output(model, name::Symbol, variables)
    cache = model.ext[:state_out]::StateCache
    init_state(cache, name, variables)
    return nothing
end

"""
"""
function init_state(cache::StateCache, name::Symbol, variables)
    len = length(variables)
    dimension = length(cache.variables)
    first = dimension + 1
    sizehint!(cache.variables, dimension + len)
    for i in eachindex(variables)
        push!(cache.variables, variables[i])
    end
    push!(cache.state, StateDef(name, first, dimension))
    return nothing
end

"""
"""
function init_state(cache::StateCache, name::Symbol, variables::JuMP.VariableRef)
    len = 1 # length(variables)
    dimension = length(cache.variables)
    first = dimension + 1
    sizehint!(cache.variables, dimension + len)
    push!(cache.variables, variables)
    push!(cache.state, StateDef(name, first, dimension))
    return nothing
end

"""
"""
function check_state_match(in::StateCache, out::StateCache)
    for i in eachindex(in.state)
        if in.state[i].name != out.state[i].name
            error("State names do not match, [in] $(in.state[i].name) != $(out.state[i].name) [out]")
        end
        if in.state[i].len != out.state[i].len
            error("State lengths do not match, [in] $(in.state[i].len) != $(out.state[i].len) [out]")
        end
    end
    return nothing
end

"""
"""
function check_state_match(state_cache_in, state_cache_out, t)
    check_state_match(state_cache_in[t], state_cache_out[t-1])
end

"""
"""
function get_state(model)
    cache = model.ext[:state_out]::StateCache
    state = Vector{Float64}(undef, length(cache.variables))
    for i in eachindex(cache.variables)
        state[i] = JuMP.value(cache.variables[i])
    end
    return state
end

"""
"""
function set_state(model, state)
    cache = model.ext[:state_in]::StateCache
    if length(state) == 0
        append!(state, fill(0.0, length(cache.variables)))
    end
    for i in eachindex(cache.variables)
        JuMP.fix(cache.variables[i], state[i]; force = true)
    end
    return nothing
end