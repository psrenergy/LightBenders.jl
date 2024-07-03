
"""
    CutPoolBackwardPhase

Cut pool to store all cuts calculated in the backward phase. In this phase we store one cut per opening.
"""
Base.@kwdef mutable struct CutPoolBackwardPhase <: AbstractCutPool
    coefs::Vector{Vector{Float64}} = Vector{Float64}[]
    state::Vector{Vector{Float64}} = Vector{Float64}[]
    rhs::Vector{Float64} = Float64[]
    obj::Vector{Float64} = Float64[]
end

function number_of_cuts(pool::CutPoolBackwardPhase)
    return length(pool.rhs)
end

function store_cut!(
    pool::CutPoolBackwardPhase, 
    coefs::Vector{Float64}, 
    state::Vector{Float64}, 
    rhs::Float64, 
    obj::Float64
)
    push!(pool.coefs, coefs)
    push!(pool.state, state)
    push!(pool.rhs, rhs)
    push!(pool.obj, obj)
    return nothing
end