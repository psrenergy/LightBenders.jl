
"""
    LocalCutPool

Cut pool to store all cuts calculated in the second stage. In this phase we store one cut per scenario.
"""
Base.@kwdef mutable struct LocalCutPool <: AbstractCutPool
    coefs::Vector{Vector{Float64}} = Vector{Float64}[]
    state::Vector{Vector{Float64}} = Vector{Float64}[]
    rhs::Vector{Float64} = Float64[]
    obj::Vector{Float64} = Float64[]
end

"""
    LocalCutPool(num_scenarios::Int)

Create a pre-allocated LocalCutPool for parallel implementations where cuts
arrive out of order and need to be stored at specific scenario indices.

All fields are initialized with default values (empty vectors for coefs/state,
zeros for rhs/obj) to avoid undefined behavior if scenarios fail to complete.
Use `validate_all_scenarios_processed` to verify all scenarios have been filled.
"""
function LocalCutPool(num_scenarios::Int)
    return LocalCutPool(
        coefs = [Float64[] for _ in 1:num_scenarios],
        state = [Float64[] for _ in 1:num_scenarios],
        rhs = zeros(Float64, num_scenarios),
        obj = zeros(Float64, num_scenarios),
    )
end

function number_of_cuts(pool::LocalCutPool)
    return length(pool.rhs)
end

"""
    validate_all_scenarios_processed(pool::LocalCutPool, num_scenarios::Int)

Validate that all scenario indices in a pre-allocated LocalCutPool have been filled.
This should be called before using the pool in parallel implementations to ensure
no worker failures or incomplete processing occurred.

Throws an error if any scenario has not been processed (detected by empty coefs vector).
"""
function validate_all_scenarios_processed(pool::LocalCutPool, num_scenarios::Int)
    for s in 1:num_scenarios
        if isempty(pool.coefs[s])
            error("Scenario $s was not processed - cut pool is incomplete. This may indicate a worker failure or error in parallel execution.")
        end
    end
    return nothing
end

function store_cut!(
    pool::LocalCutPool,
    coefs::Vector{Float64},
    state::Vector{Float64},
    rhs::Float64,
    obj::Float64,
)
    push!(pool.coefs, coefs)
    push!(pool.state, state)
    push!(pool.rhs, rhs)
    push!(pool.obj, obj)
    return nothing
end

"""
    store_cut!(pool::LocalCutPool, coefs, state, rhs, obj, scenario::Int)

Store a cut at a specific scenario index. This is used in parallel implementations
where cuts may arrive out of order. The pool must be pre-allocated with the correct size
using `LocalCutPool(num_scenarios)`.
"""
function store_cut!(
    pool::LocalCutPool,
    coefs::Vector{Float64},
    state::Vector{Float64},
    rhs::Float64,
    obj::Float64,
    scenario::Int,
)
    pool.coefs[scenario] = coefs
    pool.state[scenario] = state
    pool.rhs[scenario] = rhs
    pool.obj[scenario] = obj
    return nothing
end
