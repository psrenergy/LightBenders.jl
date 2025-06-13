Base.@kwdef mutable struct MIPOptions
    run_mip_after_iteration::Int = 50
    # TODO: Implement this
    fix_variables::Bool = false
    fix_variables_step::Int = 10
end