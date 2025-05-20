struct StateDef
    name::Symbol
    first::Int
    len::Int
end

Base.@kwdef mutable struct StateCache
    variables::Vector{JuMP.VariableRef} = JuMP.VariableRef[]
    state::Vector{StateDef} = StateDef[]
end

function set_first_stage_state(model, name::Symbol, variables)
    cache = model.ext[:first_stage_state]::StateCache
    init_state(cache, name, variables)
    return nothing
end

function set_second_stage_state(model, name::Symbol, variables)
    cache = model.ext[:second_stage_state]::StateCache
    init_state(cache, name, variables)
    return nothing
end

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

function init_state(cache::StateCache, name::Symbol, variables::JuMP.VariableRef)
    len = 1 # length(variables)
    dimension = length(cache.variables)
    first = dimension + 1
    sizehint!(cache.variables, dimension + len)
    push!(cache.variables, variables)
    push!(cache.state, StateDef(name, first, dimension))
    return nothing
end

function check_state_match(first_stage_state::StateCache, second_stage_state::StateCache)
    for i in eachindex(first_stage_state.state)
        if first_stage_state.state[i].name != second_stage_state.state[i].name
            error(
                "State names do not match, [first stage state] $(first_stage_state.state[i].name) != $(second_stage_state.state[i].name) [second stage state]",
            )
        end
        if first_stage_state.state[i].len != second_stage_state.state[i].len
            error(
                "State lengths do not match, [first stage state] $(first_stage_state.state[i].len) != $(second_stage_state.state[i].len) [second stage state]",
            )
        end
    end
    return nothing
end

function get_state(model)
    cache = model.ext[:first_stage_state]::StateCache
    state = Vector{Float64}(undef, length(cache.variables))
    for i in eachindex(cache.variables)
        state[i] = JuMP.value(cache.variables[i])
    end
    return state
end

function set_state(model, state)
    cache = model.ext[:second_stage_state]::StateCache
    if length(state) == 0
        append!(state, fill(0.0, length(cache.variables)))
    end
    for i in eachindex(cache.variables)
        if JuMP.is_parameter(cache.variables[i])
            JuMP.set_parameter_value(cache.variables[i], state[i])
        else
            JuMP.fix(cache.variables[i], state[i]; force = true)
        end
    end
    return nothing
end
