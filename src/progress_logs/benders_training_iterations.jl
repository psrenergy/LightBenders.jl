Base.@kwdef mutable struct BendersTrainingIterationsLog <: AbstractProgressLog
    LB::Vector{Float64} = []
    UB::Vector{Float64} = []
    current_iteration::Int = 0
    start_time::Float64 = time()
    progress_table::ProgressTable
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

    progress_table = ProgressTable(
        header = ["Iteration", "Lower bound", "Upper bound", "Gap", "Time [s]"],
        widths = [11, 13, 13, 13, 11],
        format = ["%d", "%.4e", "%.4e", "%.4e", "%.2f"],
        border = true,
        color = [:normal, :normal, :normal, :light_magenta, :normal],
        alignment = [:right, :center, :center, :center, :right],
    )
    initialize(progress_table)

    return BendersTrainingIterationsLog(progress_table = progress_table)
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
    next(progress.progress_table, 
        [
            progress.current_iteration,
            current_lower_bound(progress),
            current_upper_bound(progress),
            current_gap(progress),
            time() - progress.start_time,
        ]
    )
end

function finish_training!(progress::BendersTrainingIterationsLog)
    finalize(progress.progress_table)
end