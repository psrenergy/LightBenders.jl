Base.@kwdef mutable struct BendersTrainingIterationsLog <: AbstractProgressLog
    LB::Vector{Float64} = []
    UB::Vector{Float64} = []
    current_iteration::Int = 0
    start_time::Float64 = time()
    time_iteration::Vector{Float64} = Float64[]
    progress_table::ProgressTable
end

function BendersTrainingIterationsLog(policy_training_options::PolicyTrainingOptions)
    if policy_training_options.verbose
        @info(" ")
        @info("Benders Training")
        @info(" ")
        @info("Training options:")
        @info("Number of scenarios: " * string(policy_training_options.num_scenarios))
        @info("Cut strategy: " * string(policy_training_options.cut_strategy))
        @info("Risk measure: " * string(policy_training_options.risk_measure))
        @info("Stopping rule: " * string(policy_training_options.stopping_rule))
    end

    progress_table = ProgressTable(
        header = ["Iteration", "Lower bound", "Upper bound", "Gap", "Time [s]"],
        widths = [11, 13, 13, 13, 11],
        format = ["%d", "%.4e", "%.4e", "%.4e", "%.2f"],
        border = true,
        color = [:normal, :normal, :normal, :light_magenta, :normal],
        alignment = [:right, :center, :center, :center, :right],
    )
    if policy_training_options.verbose
        initialize(progress_table)
    end

    return BendersTrainingIterationsLog(progress_table = progress_table)
end

function current_time(progress::BendersTrainingIterationsLog)
    return progress.time_iteration[progress.current_iteration]
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
    push!(progress.time_iteration, 0.0)
    progress.current_iteration += 1
    return nothing
end

function report_current_bounds(progress::BendersTrainingIterationsLog)
    next(progress.progress_table,
        [
            progress.current_iteration,
            current_lower_bound(progress),
            current_upper_bound(progress),
            current_gap(progress),
            current_time(progress),
        ],
    )
    return nothing
end

function finish_training!(
    progress::BendersTrainingIterationsLog,
    convergence_result::ConvergenceResult,
)
    finalize(progress.progress_table)
    @info(results_message(convergence_result))
    return nothing
end
