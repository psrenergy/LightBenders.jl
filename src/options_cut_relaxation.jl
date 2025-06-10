Base.@kwdef mutable struct CutRelaxationOptions
    active::Bool = true
    tol::Float64 = 1e-4
    step_size::Int = 4
    warmstart::Bool = true
    warmstart_size::Int = 4
end
