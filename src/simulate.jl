abstract type AbstractSimulationImplementation end

struct BendersSerialSimulation <: AbstractSimulationImplementation end

""" 
"""
@enumx SimulationStateHandling begin
    # This is also known as commercial simulation
    StatesFixedInPolicyResult = 0
    StatesRecalculatedInSimulation = 1
end

"""
"""
Base.@kwdef mutable struct SimulationOptions
    num_stages::Int
    # TODO in the near future we should allow users to select the scenarios
    num_scenarios::Int
    gather_outputs::Bool = true
    outputs_path::AbstractString = ""
    implementation_strategy::AbstractSimulationImplementation = BendersSerialSimulation()
    state_handling::SimulationStateHandling.T = SimulationStateHandling.StatesRecalculatedInSimulation
end

function SimulationOptions(policy_training_options::PolicyTrainingOptions; kwargs...)
    return SimulationOptions(;
        num_stages = policy_training_options.num_stages,
        num_scenarios = policy_training_options.num_scenarios,
        kwargs...
    )
end

"""
"""
function simulate(;
    model_builder::Function,
    model_modifier::Function,
    results_recorder::Union{Function, Nothing} = nothing,
    inputs = nothing,
    policy::Policy,
    simulation_options::SimulationOptions
)::Float64
    if simulation_options.implementation_strategy isa BendersSerialSimulation
        return serial_benders_simulate(;model_builder, model_modifier, results_recorder, inputs, policy, simulation_options)
    else
        error("Algorithm not implemented.")
    end
end

