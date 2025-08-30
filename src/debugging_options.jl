Base.@kwdef mutable struct DebuggingOptions
    logs_dir::String = ""
    write_lp::Bool = false
    callback::Union{Function, Nothing} = nothing
end
