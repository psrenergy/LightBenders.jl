module LightBenders

using EnumX
using JobQueueMPI
using JuMP
using LinearAlgebra
using ProgressTables
using Statistics

const JQM = JobQueueMPI

# Keys aspects of the algorithm
include("debugging_options.jl")
include("results.jl")
include("states.jl")
include("risk_measures.jl")
include("options_retry.jl")

# Cut strategy implementations
include("cut_strategies/cuts_base.jl")
include("cut_strategies/local_cuts.jl")
include("cut_strategies/single_cut.jl")
include("cut_strategies/multi_cut.jl")

# Stopping rule implementations
include("stopping_rules.jl")

# Basic interfaces for training
include("train.jl")

# Progress tracking
include("progress_logs/abstractions.jl")
include("progress_logs/benders_training_iterations.jl")
include("progress_logs/deterministic_equivalent.jl")

# Interfaces for training results and simulation
include("policy.jl")
include("simulate.jl")

# training implementations
include("training_strategies/benders_serial.jl")
include("training_strategies/benders_job_queue.jl")

# simulation implementations
include("simulation_strategies/benders_serial.jl")

# deterministic equivalent formulation
include("deterministic_equivalent/deterministic_equivalent.jl")
include("deterministic_equivalent/results.jl")

# Helpers
include("optimize_helpers.jl")

# Utilities
include("utils.jl")

end # module LightBenders
