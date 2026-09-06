"""
    StateDef

One block of state variables. `scenario == 0` is a state shared by every
scenario (the classic Benders state). In the first stage `scenario == s > 0`
marks a block that only the subproblem of scenario `s` receives (e.g. per
scenario set-points implied by a decision rule); in the second stage
`scenario == CURRENT_SCENARIO` marks the block that receives the current
scenario's slice.
"""
struct StateDef
    name::Symbol
    first::Int
    len::Int
    scenario::Int
end

StateDef(name::Symbol, first::Int, len::Int) = StateDef(name, first, len, 0)

const CURRENT_SCENARIO = -1

Base.@kwdef mutable struct StateCache
    variables::Vector{JuMP.VariableRef} = JuMP.VariableRef[]
    state::Vector{StateDef} = StateDef[]
    # scenario tag of each entry of `variables` (see StateDef)
    scenario::Vector{Int} = Int[]
end

"""
    set_first_stage_state(model, name, variables; scenario = 0)

Register first stage state variables. `scenario = s > 0` registers a block that
only the subproblem of scenario `s` sees; the second stage registers the
matching block once with `scenario = CURRENT_SCENARIO`.
"""
function set_first_stage_state(model, name::Symbol, variables; scenario::Int = 0)
    cache = model.ext[:first_stage_state]::StateCache
    init_state(cache, name, variables, scenario)
    return nothing
end

function set_second_stage_state(model, name::Symbol, variables; scenario::Int = 0)
    cache = model.ext[:second_stage_state]::StateCache
    init_state(cache, name, variables, scenario)
    return nothing
end

function init_state(cache::StateCache, name::Symbol, variables, scenario::Int = 0)
    len = length(variables)
    dimension = length(cache.variables)
    first = dimension + 1
    sizehint!(cache.variables, dimension + len)
    for i in eachindex(variables)
        push!(cache.variables, variables[i])
        push!(cache.scenario, scenario)
    end
    push!(cache.state, StateDef(name, first, dimension, scenario))
    return nothing
end

function init_state(cache::StateCache, name::Symbol, variables::JuMP.VariableRef, scenario::Int = 0)
    len = 1 # length(variables)
    dimension = length(cache.variables)
    first = dimension + 1
    sizehint!(cache.variables, dimension + len)
    push!(cache.variables, variables)
    push!(cache.scenario, scenario)
    push!(cache.state, StateDef(name, first, dimension, scenario))
    return nothing
end

"""
    has_scenario_states(cache::StateCache)

Whether any registered state block is scenario specific.
"""
has_scenario_states(cache::StateCache) = any(!=(0), cache.scenario)

"""
    state_indices(first_stage_cache::StateCache, scenario::Int)

Indices (into the full first stage state vector) of the entries the subproblem
of `scenario` receives: the shared ones plus the block tagged with `scenario`.
Returns `1:n` when no block is scenario specific.
"""
function state_indices(cache::StateCache, scenario::Int)
    n = length(cache.variables)
    has_scenario_states(cache) || return collect(1:n)
    return [i for i in 1:n if cache.scenario[i] == 0 || cache.scenario[i] == scenario]
end

"""
    scenario_state(first_stage_cache, state, scenario)

Slice of the full `state` vector that the subproblem of `scenario` receives.
"""
function scenario_state(cache::StateCache, state::Vector{Float64}, scenario::Int)
    has_scenario_states(cache) || return state
    return state[state_indices(cache, scenario)]
end

"""
    expand_cut_coefficients(first_stage_cache, coefs, scenario)

Map cut coefficients computed on a scenario slice back to the full first stage
state space (sparse). Dense pass-through when no block is scenario specific.
"""
function expand_cut_coefficients(cache::StateCache, coefs::Vector{Float64}, scenario::Int)
    has_scenario_states(cache) || return coefs
    idx = state_indices(cache, scenario)
    length(idx) == length(coefs) || error(
        "Cut has $(length(coefs)) coefficients but scenario $scenario receives $(length(idx)) states",
    )
    return SparseArrays.sparsevec(idx, coefs, length(cache.variables))
end

function _state_defs_for_scenario(cache::StateCache, scenario::Int)
    return [d for d in cache.state if d.scenario == 0 || d.scenario == scenario]
end

function check_state_match(first_stage_state::StateCache, second_stage_state::StateCache)
    if !has_scenario_states(first_stage_state)
        return _check_state_defs_match(first_stage_state.state, second_stage_state.state)
    end
    for d in second_stage_state.state
        if d.scenario != 0 && d.scenario != CURRENT_SCENARIO
            error("Second stage state $(d.name) must be tagged 0 (shared) or CURRENT_SCENARIO, got $(d.scenario)")
        end
    end
    scenarios = sort(unique(s for s in first_stage_state.scenario if s > 0))
    for s in scenarios
        _check_state_defs_match(_state_defs_for_scenario(first_stage_state, s), second_stage_state.state)
    end
    return nothing
end

function _check_state_defs_match(first_defs::Vector{StateDef}, second_defs::Vector{StateDef})
    if length(first_defs) != length(second_defs)
        error(
            "Number of state blocks does not match, [first stage] $(length(first_defs)) != $(length(second_defs)) [second stage]",
        )
    end
    for i in eachindex(first_defs)
        if first_defs[i].name != second_defs[i].name
            error(
                "State names do not match, [first stage state] $(first_defs[i].name) != $(second_defs[i].name) [second stage state]",
            )
        end
        if first_defs[i].len != second_defs[i].len
            error(
                "State lengths do not match, [first stage state] $(first_defs[i].len) != $(second_defs[i].len) [second stage state]",
            )
        end
    end
    return nothing
end

function get_state(model)
    cache = model.ext[:first_stage_state]::StateCache
    state = Vector{Float64}(undef, length(cache.variables))
    for i in eachindex(cache.variables)
        value = JuMP.value(cache.variables[i])
        # Ensure the state variable is within bounds
        if has_upper_bound(cache.variables[i])
            if value > JuMP.upper_bound(cache.variables[i])
                value = JuMP.upper_bound(cache.variables[i])
            end
        end
        if has_lower_bound(cache.variables[i])
            if value < JuMP.lower_bound(cache.variables[i])
                value = JuMP.lower_bound(cache.variables[i])
            end
        end
        if is_binary(cache.variables[i])
            if value > 0.5
                value = 1.0
            else
                value = 0.0
            end
        end
        state[i] = value
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
