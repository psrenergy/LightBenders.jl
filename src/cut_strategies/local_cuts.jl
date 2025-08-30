
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
