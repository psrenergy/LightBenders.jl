
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
"""
function LocalCutPool(num_scenarios::Int)
    return LocalCutPool(
        coefs = Vector{Vector{Float64}}(undef, num_scenarios),
        state = Vector{Vector{Float64}}(undef, num_scenarios),
        rhs = Vector{Float64}(undef, num_scenarios),
        obj = Vector{Float64}(undef, num_scenarios),
    )
end

function number_of_cuts(pool::LocalCutPool)
    return length(pool.rhs)
end

function store_cut!(
    pool::LocalCutPool,
    coefs::Vector{Float64},
    state::Vector{Float64},
    rhs::Float64,
    obj::Float64,
)
    if !cut_is_different(pool, coefs, state, rhs, obj)
        return nothing
    end
    push!(pool.coefs, coefs)
    push!(pool.state, state)
    push!(pool.rhs, rhs)
    push!(pool.obj, obj)
    return nothing
end

"""
    store_cut!(pool::LocalCutPool, coefs, state, rhs, obj, scenario::Int)

Store a cut at a specific scenario index. This is used in parallel implementations
where cuts may arrive out of order. The pool must be pre-allocated with the correct size.
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
