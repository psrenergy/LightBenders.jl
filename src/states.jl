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
function set_state(model, name::Symbol, variables)
    cache = model.ext[:state]::StateCache
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
function get_state(model)
    cache = model.ext[:state]::StateCache
    state = Vector{Float64}(undef, length(cache.variables))
    for i in eachindex(cache.variables)
        state[i] = JuMP.value(cache.variables[i])
    end
    return state
end

"""
"""
function set_state(model, state)
    cache = model.ext[:state]::StateCache
    if length(state) == 0
        append!(state, fill(0.0, length(cache.variables)))
    end
    for i in eachindex(cache.variables)
        JuMP.fix(cache.variables[i], state[i]; force = true)
    end
    return nothing
end