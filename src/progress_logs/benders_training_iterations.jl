Base.@kwdef mutable struct BendersTrainingIterationsLog <: AbstractProgressLog
    LB::Vector{Float64} = Float64[]
    UB::Vector{Float64} = Float64[]
    current_iteration::Int = 0
    start_time::Float64 = time()
end

function BendersTrainingIterationsLog(policy_training_options::PolicyTrainingOptions)
    println(" ")
    println("Benders Training")
    println(" ")
    println("Training options:")
    println("Number of stages: 2")
    println("Number of scenarios: ", policy_training_options.num_scenarios)
    println("Cut strategy: ", policy_training_options.cut_strategy)
    println("Risk measure: ", policy_training_options.risk_measure)
    println("Stopping rule: ", policy_training_options.stopping_rule)
    # TODO add more prints
    println("-------------------------------------------------------------------")
    println("  iteration     lower bound     upper bound      gap     time (s)  ")
    println("-------------------------------------------------------------------")
    return BendersTrainingIterationsLog()
end

function current_upper_bound(progress::BendersTrainingIterationsLog)
    return progress.UB[progress.current_iteration]
end

function current_lower_bound(progress::BendersTrainingIterationsLog)
    return progress.LB[progress.current_iteration]
end

function current_gap(progress::BendersTrainingIterationsLog)
    return current_upper_bound(progress) - current_lower_bound(progress)
end

function last_upper_bound(progress::BendersTrainingIterationsLog)
    return progress.UB[end]
end

function last_lower_bound(progress::BendersTrainingIterationsLog)
    return progress.LB[end]
end

function start_iteration!(progress::BendersTrainingIterationsLog)
    push!(progress.LB, 0.0)
    push!(progress.UB, 0.0)
    progress.current_iteration += 1
end

function report_current_bounds(progress::BendersTrainingIterationsLog)
    println(
        "  ", 
        lpad(progress.current_iteration, 9), 
        "     ", 
        lpad(round(current_lower_bound(progress), digits=2), 11), 
        "     ", 
        lpad(round(current_upper_bound(progress), digits=2), 11), 
        "    ", 
        lpad(round(current_gap(progress), digits=2), 5),
        "     ",
        lpad(round(time() - progress.start_time, digits=2), 8)
    )
end

function finish_training!(progress::BendersTrainingIterationsLog)
    println("-------------------------------------------------------------------")
end